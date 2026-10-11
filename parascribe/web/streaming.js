// Browser-side energy endpointing, not a neural VAD or speaker diarizer.
// Keeps a short pre-roll; silence alone never becomes a submitted sentence.
export class SentenceGate {
  constructor({threshold = 0.015, pause = 0.8, maxSeconds = 20} = {}) {
    Object.assign(this, {threshold, pause, maxSeconds});
    this.reset();
  }
  reset() {
    this.active = false;
    this.pre = [];
    this.preSamples = this.total = this.silence = this.voiced = 0;
  }
  push(buffer) {
    const pcm = new DataView(buffer), count = buffer.byteLength / 2;
    let energy = 0;
    for (let i = 0; i < buffer.byteLength; i += 2) energy += (pcm.getInt16(i, true) / 32768) ** 2;
    const speech = Math.sqrt(energy / Math.max(1, count)) >= this.threshold;
    let frames = [];
    if (!this.active) {
      this.pre.push(buffer); this.preSamples += count;
      while (this.preSamples > 3200 + count) this.preSamples -= this.pre.shift().byteLength / 2;
      if (!speech) return {frames, boundary: null};
      this.active = true;
      frames = this.pre; this.total = this.preSamples;
      this.pre = []; this.preSamples = 0;
    } else { frames = [buffer]; this.total += count; }
    this.voiced += speech ? count : 0;
    this.silence = speech ? 0 : this.silence + count;
    let boundary = null;
    if (this.silence >= this.pause * 16000 || this.total >= this.maxSeconds * 16000) {
      boundary = this.voiced >= 4000 ? "commit" : "discard";
      this.reset();
    }
    return {frames, boundary};
  }
}
