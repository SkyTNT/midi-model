const MIDI_OUTPUT_BATCH_SIZE=4;
//Do not change MIDI_OUTPUT_BATCH_SIZE. It will be automatically replaced.

/**
 * 自动绕过 shadowRoot 的 querySelector
 * @param {string} selector - 要查询的 CSS 选择器
 * @returns {Element|null} - 匹配的元素或 null 如果未找到
 */
function deepQuerySelector(selector) {
  /**
   * 在指定的根元素或文档对象下深度查询元素
   * @param {Element|Document} root - 要开始搜索的根元素或文档对象
   * @param {string} selector - 要查询的 CSS 选择器
   * @returns {Element|null} - 匹配的元素或 null 如果未找到
   */
  function deepSearch(root, selector) {
    // 在当前根元素下查找
    let element = root.querySelector(selector);
    if (element) {
      return element;
    }

    // 如果未找到，递归检查 shadow DOM
    const shadowHosts = root.querySelectorAll('*');

    for (let i = 0; i < shadowHosts.length; i++) {
      const host = shadowHosts[i];

      // 检查当前元素是否有 shadowRoot
      if (host.shadowRoot) {
        element = deepSearch(host.shadowRoot, selector);
        if (element) {
          return element;
        }
      }
    }
    // 未找到元素
    return null;
  }

  return deepSearch(this, selector);
}

Element.prototype.deepQuerySelector = deepQuerySelector;
Document.prototype.deepQuerySelector = deepQuerySelector;

function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app')
    const gradioShadowRoot = elems.length == 0 ? null : elems[0].shadowRoot
    return !!gradioShadowRoot ? gradioShadowRoot : document;
}

uiUpdateCallbacks = []
msgReceiveCallbacks = []

function onUiUpdate(callback){
    uiUpdateCallbacks.push(callback)
}

function onMsgReceive(callback){
    msgReceiveCallbacks.push(callback)
}

function runCallback(x, m){
    try {
        x(m)
    } catch (e) {
        (console.error || console.log).call(console, e.message, e);
    }
}
function executeCallbacks(queue, m) {
    queue.forEach(function(x){runCallback(x, m)})
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        executeCallbacks(uiUpdateCallbacks, m);
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })
});

function HSVtoRGB(h, s, v) {
    let r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    };
}

function isMobile(){
  return /(iPhone|iPad|iPod|iOS|Android|Windows Phone)/i.test(navigator.userAgent);
}

const number2patch = ['Acoustic Grand', 'Bright Acoustic', 'Electric Grand', 'Honky-Tonk', 'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clav', 'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer', 'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion', 'Acoustic Guitar(nylon)', 'Acoustic Guitar(steel)', 'Electric Guitar(jazz)', 'Electric Guitar(clean)', 'Electric Guitar(muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar Harmonics', 'Acoustic Bass', 'Electric Bass(finger)', 'Electric Bass(pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2', 'SynthStrings 1', 'SynthStrings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn', 'Brass Section', 'SynthBrass 1', 'SynthBrass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown Bottle', 'Skakuhachi', 'Whistle', 'Ocarina', 'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass+lead)', 'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bagpipe', 'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot']
const number2drum_kits = {0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz", 40: "Blush", 48: "Orchestra"}

class MidiVisualizer extends HTMLElement{
    constructor() {
        super();
        this.midiEvents = [];
        this.activeNotes = [];
        this.midiTimes = [];
        this.trackMap = new Map()
        this.patches = [];
        for (let i=0;i<16;i++){
            this.patches.push([[0,0]])
        }
        this.container = null;
        this.trackList = null
        this.pianoRoll = null;
        this.svg = null;
        this.timeLine = null;
        this.config = {
            noteHeight : 4,
            beatWidth: 32
        }
        if (isMobile()){
            this.config.noteHeight = 1;
            this.config.beatWidth = 16;
        }
        this.timePreBeat = 16
        this.svgWidth = 0;
        this.t1 = 0;
        this.totalTimeMs = 0
        this.playTime = 0
        this.playTimeMs = 0
        this.lastUpdateTime = 0
        this.colorMap = new Map();
        this.playing = false;
        this.timer = null;
        this.version = "v2"
        this.init();
    }

    init(){
        this.innerHTML=''
        const shadow = this.attachShadow({mode: 'open'});
        const style = document.createElement("style");
        style.textContent = ".note.active {stroke: black;stroke-width: 0.75;stroke-opacity: 0.75;}";
        const container = document.createElement('div');
        container.style.display="flex";
        container.style.height=`${this.config.noteHeight*128 + 25}px`;
        const trackListContainer = document.createElement('div');
        trackListContainer.style.width = "260px";
        trackListContainer.style.minWidth = "260px";
        trackListContainer.style.height = "100%";
        trackListContainer.style.display="flex";
        trackListContainer.style.flexDirection="column";
        const trackList = document.createElement('div');
        trackList.style.width = "100%";
        trackList.style.height = "100%";
        trackList.style.overflowY= "scroll";
        trackList.style.display="flex";
        trackList.style.flexDirection="column";
        trackList.style.flexGrow="1";
        const trackControls = document.createElement('div');
        trackControls.style.display="flex";
        trackControls.style.flexDirection="row";
        trackControls.style.width = "100%";
        trackControls.style.height = "50px";
        trackControls.style.minHeight = "50px";
        const allTrackBtn = document.createElement('button');
        allTrackBtn.textContent = "All";
        allTrackBtn.style.width = "50%";
        allTrackBtn.style.height = "100%";
        allTrackBtn.style.backgroundColor = "rgba(200, 200, 200, 0.3)";
        allTrackBtn.style.color = 'inherit';
        allTrackBtn.style.border = "none";
        allTrackBtn.style.cursor = 'pointer';
        let self = this;
        allTrackBtn.onclick = function (){
            self.trackMap.forEach((track, id) => {
                track.setChecked(true);
            })
        };
        const noneTrackBtn = document.createElement('button');
        noneTrackBtn.textContent = "None";
        noneTrackBtn.style.width = "50%";
        noneTrackBtn.style.height = "100%";
        noneTrackBtn.style.backgroundColor = "rgba(200, 200, 200, 0.3)";
        noneTrackBtn.style.color = 'inherit';
        noneTrackBtn.style.border = "none";
        noneTrackBtn.style.cursor = 'pointer';
        noneTrackBtn.onclick = function (){
            self.trackMap.forEach((track, id) => {
                track.setChecked(false);
            });
        };
        const pianoRoll = document.createElement('div');
        pianoRoll.style.overflowX= "scroll";
        pianoRoll.style.flexGrow="1";
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.style.height = `${this.config.noteHeight*128}px`;
        svg.style.width = `${this.svgWidth}px`;
        const timeLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        timeLine.style.stroke = "green"
        timeLine.style.strokeWidth = "2";

        if (isMobile()){
            trackListContainer.style.display = "none";
            timeLine.style.strokeWidth = "1";
        }
        shadow.appendChild(style)
        shadow.appendChild(container);
        container.appendChild(trackListContainer);
        trackListContainer.appendChild(trackList);
        trackListContainer.appendChild(trackControls);
        trackControls.appendChild(allTrackBtn);
        trackControls.appendChild(noneTrackBtn);
        container.appendChild(pianoRoll);
        pianoRoll.appendChild(svg);
        svg.appendChild(timeLine)
        this.container = container;
        this.trackList = trackList;
        this.pianoRoll = pianoRoll;
        this.svg = svg;
        this.timeLine= timeLine;
        for(let i = 0; i < 128 ; i++){
            this.colorMap.set(i, HSVtoRGB(i / 128, 1, 1))
        }
        this.setPlayTime(0);
    }

    addTrack(id, tr, cl, name, color){
        const track = {id, tr, cl, name, color, empty: true,
            lastCC: new Map(),
            instrument: cl===9?"Standard Drum":"Acoustic Grand",
            svg: document.createElementNS('http://www.w3.org/2000/svg', 'g'),
            ccPaths: new Map()
        }
        this.svg.appendChild(track.svg)
        const trackItem = this.createTrackItem(track);
        this.trackList.appendChild(trackItem);
        this.trackMap.set(id, track);
        return track;
    }

    getTrack(tr, cl){
        const id = tr * 16 + cl
        let track = this.trackMap.get(id)
        if (!!track){
            return track
        }
        let color = this.colorMap.get((this.trackMap.size*53)%128)
        return this.addTrack(id, tr, cl, `Track ${tr}, Channel ${cl}`, color)
    }

    createTrackItem(track) {
        const trackItem = document.createElement('div');
        trackItem.style.display = 'flex';
        trackItem.style.alignItems = 'center';
        trackItem.style.width = '100%';
        trackItem.style.position = 'relative';

        const colorBar = document.createElement('div');
        colorBar.style.width = '5%';
        colorBar.style.height = '100%';
        colorBar.style.position = 'absolute';
        colorBar.style.left = '0';
        colorBar.style.top = '0';
        let color = track.color;
        colorBar.style.backgroundColor = `rgb(${color.r}, ${color.g}, ${color.b})`;
        trackItem.appendChild(colorBar);

        const content = document.createElement('div');
        content.style.paddingLeft = '30px';
        content.style.flexGrow = '1';
        content.style.color = "grey"
        content.innerHTML = `<p>${track.name}<br>${track.instrument}</p>`;
        trackItem.appendChild(content);
        track.updateInstrument = function (instrument){
            track.instrument = instrument;
            content.innerHTML = `<p>${track.name}<br>${track.instrument}</p>`;
        }
        track.setEmpty = function (empty){
            if (empty!==track.empty){
                content.style.color = empty?"grey":"inherit";
            }
        }

        const toggleSwitch = document.createElement('input');
        toggleSwitch.type = 'checkbox';
        toggleSwitch.checked = true;
        toggleSwitch.style.marginLeft = 'auto';
        toggleSwitch.style.marginRight = '10px';
        toggleSwitch.style.width = '20px';
        toggleSwitch.style.height = '20px';
        toggleSwitch.style.cursor = 'pointer';

        toggleSwitch.onchange = function () {
            track.svg.setAttribute('visibility',toggleSwitch.checked? "visible" : "hidden")
        };
        track.setChecked = function (checked){
            toggleSwitch.checked = checked;
            track.svg.setAttribute('visibility',toggleSwitch.checked? "visible" : "hidden")
        }
        trackItem.appendChild(toggleSwitch);
        return trackItem;
    }

    clearMidiEvents(){
        this.pause()
        this.midiEvents = [];
        this.activeNotes = [];
        this.midiTimes = [];
        this.trackMap = new Map()
        this.patches = [];
        for (let i=0;i<16;i++){
            this.patches.push([[0,0]])
        }
        this.t1 = 0
        this.setPlayTime(0);
        this.totalTimeMs = 0;
        this.playTimeMs = 0
        this.lastUpdateTime = 0
        this.trackList.innerHTML = ''
        this.svgWidth = 0
        this.svg.innerHTML = ''
        this.svg.style.width = `${this.svgWidth}px`;
        this.svg.appendChild(this.timeLine)
    }

    appendMidiEvent(midiEvent){
        if(midiEvent instanceof Array && midiEvent.length > 0){

            this.t1 += midiEvent[1]
            let t = this.t1*this.timePreBeat + midiEvent[2]
            midiEvent = [midiEvent[0], t].concat(midiEvent.slice(3))
            if(midiEvent[0] === "note"){
                let track = midiEvent[2]
                let duration = 0
                let channel = 0
                let pitch = 0
                let velocity = 0
                if(this.version === "v1"){
                    duration = midiEvent[3]
                    channel = midiEvent[4]
                    pitch = midiEvent[5]
                    velocity = midiEvent[6]
                }else if (this.version === "v2"){
                    channel = midiEvent[3]
                    pitch = midiEvent[4]
                    velocity = midiEvent[5]
                    duration = midiEvent[6]
                }
                let vis_track = this.getTrack(track, channel);
                vis_track.setEmpty(false);
                let x = (t/this.timePreBeat)*this.config.beatWidth
                let y = (127 - pitch)*this.config.noteHeight
                let w = (duration/this.timePreBeat)*this.config.beatWidth
                let h = this.config.noteHeight
                this.svgWidth = Math.ceil(Math.max(x + w, this.svgWidth))
                let opacity = Math.min(1, velocity/127 + 0.1).toFixed(2)
                let rect = this.drawNote(vis_track, x,y,w,h, opacity)
                midiEvent.push(rect);
                this.setPlayTime(t);
                this.pianoRoll.scrollTo(this.svgWidth - this.pianoRoll.offsetWidth, this.pianoRoll.scrollTop)
            }else if(midiEvent[0] === "patch_change"){
                let track = midiEvent[2];
                let channel = midiEvent[3];
                this.patches[channel].push([t, midiEvent[4]]);
                this.patches[channel].sort((a, b) => a[0] - b[0]);
                this.getTrack(track, channel);
            }else if(midiEvent[0] === "control_change"){
                let track = midiEvent[2];
                let channel = midiEvent[3];
                let controller = midiEvent[4];
                let value = midiEvent[5];
                let vis_track = this.getTrack(track, channel);
                this.drawCC(vis_track, t, controller, value);
                this.setPlayTime(t);
            }
            this.midiEvents.push(midiEvent);
            this.svg.style.width = `${this.svgWidth}px`;
        }

    }

    drawNote(track, x, y, w, h, opacity) {
        if (!track.svg) {
          return null;
        }
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.classList.add('note');
        const color = track.color;
        rect.setAttribute('fill', `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`);
        // Round values to the nearest integer to avoid partially filled pixels.
        rect.setAttribute('x', `${Math.round(x)}`);
        rect.setAttribute('y', `${Math.round(y)}`);
        rect.setAttribute('width', `${Math.round(w)}`);
        rect.setAttribute('height', `${Math.round(h)}`);
        track.svg.appendChild(rect);
        return rect
    }

    drawCC(track, t, controller, value){
        if (!track.svg) {
          return null;
        }
        let path = track.ccPaths.get(controller);
        let x = (t/this.timePreBeat)*this.config.beatWidth
        let y = (127 - value)*this.config.noteHeight
        if (!path){
            path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('visibility',"hidden");
            path.setAttribute('fill', "transparent");
            const color = track.color;
            path.setAttribute('stroke', `rgba(${color.r}, ${color.g}, ${color.b}, 0.6)`);
            path.setAttribute('stroke-width', "1");
            path.setAttribute('d',
                t===0?`M ${x} ${y}`:`M 0 ${127*this.config.noteHeight} H ${x} V ${y}`);
            track.svg.appendChild(path);
            track.ccPaths.set(controller, path);
            track.lastCC.set(controller, value);
            return path;
        }
        let lastVal = track.lastCC.get(controller);
        if(lastVal !== value){
            path.removeAttribute('visibility');
        }
        let d = path.getAttribute("d");
        d += `H ${x} V ${y}`
        path.setAttribute('d', d);
        return path
    }

    finishAppendMidiEvent(){
        this.pause()
        let midiEvents = this.midiEvents.sort((a, b)=>a[1]-b[1])
        let tempo = (60 / 120) * 10 ** 3
        let ms = 0
        let lastT = 0
        this.midiTimes.push({ms:ms, t: 0, tempo: tempo})
        midiEvents.forEach((midiEvent)=>{
            let t = midiEvent[1]
            ms += ((t- lastT) / this.timePreBeat) * tempo
            if(midiEvent[0]==="set_tempo"){
                tempo = (60 / midiEvent[3]) * 10 ** 3
                this.midiTimes.push({ms:ms, t: t, tempo: tempo})
            }
            if(midiEvent[0]==="note"){
                this.totalTimeMs = Math.max(this.totalTimeMs, ms + (midiEvent[3]/ this.timePreBeat)*tempo)
            }else{
                this.totalTimeMs = Math.max(this.totalTimeMs, ms);
            }
            lastT = t;
        })
        let x = (lastT/this.timePreBeat)*this.config.beatWidth;
        this.trackMap.forEach((track, id)=>{
            track.ccPaths.forEach((path, controller)=>{
                let d = path.getAttribute("d");
                d += `H ${x}`
                path.setAttribute('d', d);
            })
        })
    }

    setPlayTime(t){
        this.playTime = t
        let x = Math.round((t/this.timePreBeat)*this.config.beatWidth)
        this.timeLine.setAttribute('x1', `${x}`);
        this.timeLine.setAttribute('y1', '0');
        this.timeLine.setAttribute('x2', `${x}`);
        this.timeLine.setAttribute('y2', `${this.config.noteHeight*128}`);

        this.pianoRoll.scrollTo(Math.max(0, x - this.pianoRoll.offsetWidth/2), this.pianoRoll.scrollTop)

        this.trackMap.forEach((track, id)=>{
            let instrument = track.instrument
            let cl = track.cl;
            let patches = this.patches[cl]
            let p = 0
            for (let i = 0; i < patches.length ; i++){
                let tp = patches[i]
                if (t < tp[0])
                    break
                p = tp[1]
            }
            if (cl === 9){
                let drumKit = number2drum_kits[`${p}`];
                if (!!drumKit)
                    instrument = drumKit + " Drum";
            }else{
                instrument = number2patch[p]
            }
            if (instrument !== track.instrument)
                track.updateInstrument(instrument)
        });

        let dt = Date.now() - this.lastUpdateTime; // limit the update rate of ActiveNotes
        if(this.playing && dt > 50){
            let activeNotes = []
            this.removeActiveNotes(this.activeNotes)
            this.midiEvents.forEach((midiEvent)=>{
                if(midiEvent[0] === "note"){
                    let time = midiEvent[1]
                    let duration = this.version==="v1"? midiEvent[3]:midiEvent[6]
                    let note = midiEvent[midiEvent.length - 1]
                    if(time <=this.playTime && time+duration>= this.playTime){
                        activeNotes.push(note)
                    }
                }
            });
            this.addActiveNotes(activeNotes)
            this.lastUpdateTime = Date.now();
        }

    }

    setPlayTimeMs(ms){
        this.playTimeMs = ms
        let playTime = 0
        for(let i =0;i<this.midiTimes.length;i++){
            let midiTime = this.midiTimes[i]
            if(midiTime.ms>=ms){
                break;
            }
            playTime = midiTime.t + (ms-midiTime.ms) * this.timePreBeat / midiTime.tempo
        }
        this.setPlayTime(playTime)
    }

    addActiveNotes(notes){
        notes.forEach((note)=>{
            this.activeNotes.push(note)
            note.classList.add('active');
        });
    }

    removeActiveNotes(notes){
        notes.forEach((note)=>{
            let idx = this.activeNotes.indexOf(note)
            if(idx>-1)
                this.activeNotes.splice(idx, 1);
            note.classList.remove('active');
        });
    }

    play(){
        this.playing = true;
    }

    pause(){
        this.removeActiveNotes(this.activeNotes)
        this.playing = false;
    }


    bindAudioPlayer(audio){
        this.pause()
        audio.addEventListener("play", (event)=>{
            this.play()
        })
        audio.addEventListener("pause", (event)=>{
            this.pause()
        })
        audio.addEventListener("loadedmetadata", (event)=>{
            //I don't know why the calculated totalTimeMs is different from audio.duration*10**3
            this.totalTimeMs = audio.duration*10**3;
        })
    }

    bindWaveformCursor(cursor){
        let self = this;
        const callback = function(mutationsList, observer) {
            for(let mutation of mutationsList) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    let progress = parseFloat(mutation.target.style.left.slice(0,-1))*0.01;
                    if(!isNaN(progress)){
                        self.setPlayTimeMs(progress*self.totalTimeMs);
                    }
                }
            }
        };
        const observer = new MutationObserver(callback);
        observer.observe(cursor, {
            attributes: true,
            attributeFilter: ['style']
        });
    }
}

customElements.define('midi-visualizer', MidiVisualizer);

(()=>{
    function midi_visualizer_setup(idx, midi_visualizer){
        let midi_visualizer_container_inited = null
        let midi_audio_audio_inited = null;
        let midi_audio_cursor_inited = null;
        onUiUpdate((m)=>{
            let app = gradioApp()
            let midi_visualizer_container = app.querySelector(`#midi_visualizer_container_${idx}`);
            if(!!midi_visualizer_container && midi_visualizer_container_inited!== midi_visualizer_container){
                midi_visualizer_container.appendChild(midi_visualizer)
                midi_visualizer_container_inited = midi_visualizer_container;
            }
            let midi_audio = app.querySelector(`#midi_audio_${idx}`);
            if (!!midi_audio){
                let midi_audio_cursor = midi_audio.deepQuerySelector(".cursor");
                if(!!midi_audio_cursor && midi_audio_cursor_inited!==midi_audio_cursor){
                    midi_visualizer.bindWaveformCursor(midi_audio_cursor)
                    midi_audio_cursor_inited = midi_audio_cursor
                }
                let midi_audio_waveform = midi_audio.deepQuerySelector("#waveform");
                if(!!midi_audio_waveform){
                    let midi_audio_audio = midi_audio_waveform.deepQuerySelector("audio");
                    if(!!midi_audio_audio && midi_audio_audio_inited!==midi_audio_audio){
                        midi_visualizer.bindAudioPlayer(midi_audio_audio)
                        midi_audio_audio_inited = midi_audio_audio
                    }
                }
            }
        });
    }

    let midi_visualizers = []
    for (let i = 0; i < MIDI_OUTPUT_BATCH_SIZE ; i++){
        let midi_visualizer = document.createElement('midi-visualizer');
        midi_visualizers.push(midi_visualizer);
        midi_visualizer_setup(i, midi_visualizer)
    }

    let hasProgressBar = false;
    let output_tabs_inited = null;
    onUiUpdate((m)=>{
        let app = gradioApp()
        let output_tabs = app.querySelector("#output_tabs");
        if(!!output_tabs && output_tabs_inited!== output_tabs){
            output_tabs_inited = output_tabs;
        }
    });

    function createProgressBar(progressbarContainer){
        let parentProgressbar = progressbarContainer.parentNode;
        let divProgress = document.createElement('div');
        divProgress.className='progressDiv';
        let rect = progressbarContainer.getBoundingClientRect();
        divProgress.style.width = rect.width + "px";
        divProgress.style.background = "#b4c0cc";
        divProgress.style.borderRadius = "8px";
        let divInner = document.createElement('div');
        divInner.className='progress';
        divInner.style.color = "white";
        divInner.style.background = "#0060df";
        divInner.style.textAlign = "right";
        divInner.style.fontWeight = "bold";
        divInner.style.borderRadius = "8px";
        divInner.style.height = "20px";
        divInner.style.lineHeight = "20px";
        divInner.style.paddingRight = "8px"
        divInner.style.width = "0%";
        divProgress.appendChild(divInner);
        parentProgressbar.insertBefore(divProgress, progressbarContainer);
        hasProgressBar = true;
    }

    function removeProgressBar(progressbarContainer){
        let parentProgressbar = progressbarContainer.parentNode;
        let divProgress = parentProgressbar.querySelector(".progressDiv");
        parentProgressbar.removeChild(divProgress);
        hasProgressBar = false;
    }

    function setProgressBar(progress, total){
        if (!hasProgressBar)
            createProgressBar(output_tabs_inited)
        if (hasProgressBar && total === 0){
            removeProgressBar(output_tabs_inited)
            return
        }
        let parentProgressbar = output_tabs_inited.parentNode;
        // let divProgress = parentProgressbar.querySelector(".progressDiv");
        let divInner = parentProgressbar.querySelector(".progress");
        if(total===0)
            total = 1;
        divInner.style.width = `${(progress/total)*100}%`;
        divInner.textContent = `${progress}/${total}`;
    }

    onMsgReceive((msgs)=>{
        for(let msg of msgs){
            if(msg instanceof Array){
                msg.forEach((o)=>{handleMsg(o)});
            }else{
                handleMsg(msg);
            }
        }
    })
    function handleMsg(msg){
        let idx;
        switch (msg.name) {
            case "visualizer_clear":
                idx = msg.data[0];
                let ver = msg.data[1];
                midi_visualizers[idx].clearMidiEvents(false);
                midi_visualizers[idx].version = ver;
                break;
            case "visualizer_append":
                idx = msg.data[0];
                let events = msg.data[1];
                events.forEach( value => {
                    midi_visualizers[idx].appendMidiEvent(value);
                })
                break;
            case "visualizer_end":
                idx = msg.data;
                midi_visualizers[idx].finishAppendMidiEvent()
                midi_visualizers[idx].setPlayTime(0);
                break;
            case "progress":
                let progress = msg.data[0]
                let total = msg.data[1]
                setProgressBar(progress, total)
                break;
            default:
        }
    }
})();
