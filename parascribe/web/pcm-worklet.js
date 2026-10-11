// Convert the browser's actual sample rate to mono PCM16 at 16 kHz.
// Fractional integration carries state across render blocks (including 44.1 kHz).
class Pcm16Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.weight = 0;
    this.sum = 0;
    this.samples = [];
    this.paused = false;
    this.port.onmessage = ({data}) => {
      if (data.type === "flush") {
        if (data.pause) this.paused = true;
        // Commit the fractional tail to the current utterance, never the next.
        if (this.weight > 0) this.samples.push(this.sum / this.weight);
        this.weight = this.sum = 0;
        this.flush();
        this.port.postMessage({type: "flushed", id: data.id});
      }
    };
  }

  flush() {
    if (!this.samples.length) return;
    const buffer = new ArrayBuffer(this.samples.length * 2);
    const view = new DataView(buffer);
    this.samples.forEach((sample, index) => {
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(index * 2, Math.round(clipped * (clipped < 0 ? 32768 : 32767)), true);
    });
    this.samples = [];
    this.port.postMessage({type: "audio", buffer}, [buffer]);
  }

  process(inputs) {
    const channels = inputs[0];
    if (this.paused || !channels?.length) return true;
    for (let index = 0; index < channels[0].length; index++) {
      let sample = 0;
      for (const channel of channels) sample += channel[index] / channels.length;
      let remaining = 1;
      while (remaining > 1e-9) {
        const used = Math.min(remaining, this.ratio - this.weight);
        this.sum += sample * used;
        this.weight += used;
        remaining -= used;
        if (this.weight >= this.ratio - 1e-9) {
          this.samples.push(this.sum / this.ratio);
          this.sum = this.weight = 0;
          if (this.samples.length >= 512) this.flush();
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm16-capture", Pcm16Capture);
