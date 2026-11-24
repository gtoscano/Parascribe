// Injected into a joined Jitsi Meet page.  Reads every remote audio track
// with MediaStreamTrackProcessor (raw decoded frames; no Web Audio graph or
// AudioWorklet, which does not load in headless Chromium in Docker), converts
// it to mono 16 kHz PCM16, and hands it to Python via `parascribeEmit`.
//
// Events emitted:
//   {type: "participant", id, name}                     join / rename
//   {type: "left", id}                                  participant left
//   {type: "audio", key, owner, pcm (base64), at}       owner read per chunk
//   {type: "track_ended", key}
//   {type: "conference_left", reason}
//
// Uses the Jitsi Meet web app's `APP.conference._room` (a lib-jitsi-meet
// JitsiConference).  This is an internal API; it is stable in practice but
// not a public contract.
async ({flushMs}) => {
  const emit = window.parascribeEmit;
  const room = window.APP.conference._room;
  const events = window.JitsiMeetJS?.events?.conference || {};
  const EV = {
    trackAdded: events.TRACK_ADDED || "conference.trackAdded",
    trackRemoved: events.TRACK_REMOVED || "conference.trackRemoved",
    userJoined: events.USER_JOINED || "conference.userJoined",
    userLeft: events.USER_LEFT || "conference.userLeft",
    nameChanged: events.DISPLAY_NAME_CHANGED || "conference.displayNameChanged",
    kicked: events.KICKED || "conference.kicked",
    left: events.CONFERENCE_LEFT || "conference.left",
  };

  // Mono 16 kHz by fractional averaging, as in parascribe/web/pcm-worklet.js;
  // state carries across frames so rates like 44.1 kHz stay exact.
  class Downsampler {
    constructor(rate) {
      Object.assign(this, {rate, ratio: rate / 16000, weight: 0, sum: 0});
    }
    push(planes, count) {
      const out = new Int16Array(Math.ceil(count / this.ratio) + 2);
      let produced = 0;
      for (let i = 0; i < count; i++) {
        let sample = 0;
        for (const plane of planes) sample += plane[i] / planes.length;
        let remaining = 1;
        while (remaining > 1e-9) {
          const used = Math.min(remaining, this.ratio - this.weight);
          this.sum += sample * used;
          this.weight += used;
          remaining -= used;
          if (this.weight >= this.ratio - 1e-9) {
            const value = Math.max(-1, Math.min(1, this.sum / this.ratio));
            out[produced++] = Math.round(value * (value < 0 ? 32768 : 32767));
            this.sum = this.weight = 0;
          }
        }
      }
      return out.slice(0, produced);
    }
  }

  const toBase64 = bytes => {
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
  };

  const taps = new Map();  // JitsiTrack -> tap state
  let nextKey = 1;

  function flush(tap) {
    // Send consecutive chunks with the same owner together, in order.
    while (tap.pending.length) {
      const owner = tap.pending[0].owner;
      const group = [];
      while (tap.pending.length && tap.pending[0].owner === owner) group.push(tap.pending.shift());
      const size = group.reduce((total, item) => total + item.bytes.length, 0);
      const joined = new Uint8Array(size);
      let offset = 0;
      for (const item of group) { joined.set(item.bytes, offset); offset += item.bytes.length; }
      emit({type: "audio", key: tap.key, owner, pcm: toBase64(joined), at: group.at(-1).at});
    }
  }

  function tap(track) {
    if (taps.has(track) || track.isLocal() || track.getType() !== "audio") return;
    const stream = track.getOriginalStream();
    if (!stream || !stream.getAudioTracks().length) return;
    // Keep the remote stream attached to a (muted) media element so Chrome
    // keeps decoding it regardless of how it is consumed.
    const element = new Audio();
    element.muted = true;
    element.srcObject = stream;
    element.play().catch(() => {});
    const reader = new MediaStreamTrackProcessor({track: stream.getAudioTracks()[0]})
      .readable.getReader();
    const state = {key: `t${nextKey++}`, track, element, reader, pending: [], stopped: false};
    taps.set(track, state);
    (async () => {
      let resampler = null;
      while (!state.stopped) {
        let result;
        try { result = await reader.read(); } catch { break; }
        if (result.done) break;
        const frame = result.value;
        try {
          if (resampler?.rate !== frame.sampleRate) resampler = new Downsampler(frame.sampleRate);
          const planes = [];
          for (let plane = 0; plane < frame.numberOfChannels; plane++) {
            const data = new Float32Array(frame.numberOfFrames);
            frame.copyTo(data, {planeIndex: plane, format: "f32-planar"});
            planes.push(data);
          }
          const pcm = resampler.push(planes, frame.numberOfFrames);
          if (pcm.length) {
            // Owner is read per frame: meet.jit.si may reassign a track.
            state.pending.push({owner: track.getParticipantId() || null,
                                bytes: new Uint8Array(pcm.buffer), at: Date.now() / 1000});
          }
        } finally {
          frame.close();
        }
      }
    })();
  }

  function untap(track) {
    const state = taps.get(track);
    if (!state) return;
    state.stopped = true;
    state.reader.cancel().catch(() => {});
    flush(state);
    state.element.srcObject = null;
    taps.delete(track);
    emit({type: "track_ended", key: state.key});
  }

  const announce = (id, name) => emit({type: "participant", id, name: name || ""});
  for (const participant of room.getParticipants()) {
    announce(participant.getId(), participant.getDisplayName());
    for (const track of participant.getTracks()) tap(track);
  }
  room.on(EV.userJoined, (id, user) => announce(id, user?.getDisplayName?.()));
  room.on(EV.nameChanged, (id, name) => announce(id, name));
  room.on(EV.userLeft, id => emit({type: "left", id}));
  room.on(EV.trackAdded, track => tap(track));
  room.on(EV.trackRemoved, track => untap(track));
  room.on(EV.kicked, () => emit({type: "conference_left", reason: "kicked"}));
  room.on(EV.left, () => emit({type: "conference_left", reason: "left"}));

  const timer = setInterval(() => { for (const state of taps.values()) flush(state); }, flushMs);
  window.parascribeStop = () => {
    clearInterval(timer);
    for (const track of [...taps.keys()]) untap(track);
  };
  return {tracks: taps.size};
}
