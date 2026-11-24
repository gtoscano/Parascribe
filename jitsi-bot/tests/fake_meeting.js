// A stand-in for the Jitsi Meet web app, for tests.  Installs `window.APP`
// with a JitsiConference-like room whose remote audio tracks play given PCM.
//   participants: [{id, name}]
//   tracks: [{pcm: base64 mono PCM16 16 kHz, startAt: s, owners: [[s, id|null], ...]}]
//   addAfter: s before tracks are announced; leaveAfter: s before the room ends
async ({participants, tracks, addAfter, leaveAfter}) => {
  const context = new AudioContext();
  await context.resume();
  const handlers = {};
  const fire = (event, ...args) => (handlers[event] || []).forEach(callback => callback(...args));
  const origin = performance.now() / 1000;
  const elapsed = () => performance.now() / 1000 - origin;

  const fakeTracks = tracks.map(spec => {
    const bytes = Uint8Array.from(atob(spec.pcm), c => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer);
    const buffer = context.createBuffer(1, Math.max(1, pcm.length), 16000);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    const destination = context.createMediaStreamDestination();
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(destination);
    source.start(context.currentTime + spec.startAt);
    return {
      isLocal: () => false,
      getType: () => "audio",
      getOriginalStream: () => destination.stream,
      getParticipantId: () => {
        let owner = null;
        for (const [at, id] of spec.owners) if (elapsed() >= at) owner = id;
        return owner;
      },
    };
  });

  let joined = true;
  const people = participants.map(p => ({
    getId: () => p.id, getDisplayName: () => p.name, getTracks: () => []}));
  const room = {
    on: (event, callback) => (handlers[event] ??= []).push(callback),
    getParticipants: () => (joined ? people : []),
    isJoined: () => joined,
    sendTextMessage: text => { window.fakeChat = (window.fakeChat || []).concat(text); },
  };
  window.APP = {conference: {_room: room, muteAudio: () => {}}};
  setTimeout(() => fakeTracks.forEach(track => fire("conference.trackAdded", track)),
             addAfter * 1000);
  setTimeout(() => {
    fakeTracks.forEach(track => fire("conference.trackRemoved", track));
    joined = false;
    fire("conference.left");
  }, leaveAfter * 1000);
}
