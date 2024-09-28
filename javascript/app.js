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

class MidiVisualizer extends HTMLElement{
    constructor() {
        super();
        this.midiEvents = [];
        this.activeNotes = [];
        this.midiTimes = [];
        this.wrapper = null;
        this.svg = null;
        this.timeLine = null;
        this.config = {
            noteHeight : 4,
            beatWidth: 32
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
        const wrapper = document.createElement('div');
        style.textContent = ".note.active {stroke: black;stroke-width: 0.75;stroke-opacity: 0.75;}";
        wrapper.style.overflowX= "scroll"
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.style.height = `${this.config.noteHeight*128}px`;
        svg.style.width = `${this.svgWidth}px`;
        const timeLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        timeLine.style.stroke = "green"
        timeLine.style.strokeWidth = 2;
        shadow.appendChild(style)
        shadow.appendChild(wrapper);
        wrapper.appendChild(svg);
        svg.appendChild(timeLine)
        this.wrapper = wrapper;
        this.svg = svg;
        this.timeLine= timeLine;
        this.setPlayTime(0);
    }

    clearMidiEvents(keepColor=false){
        this.pause()
        this.midiEvents = [];
        this.activeNotes = [];
        this.midiTimes = [];
        this.t1 = 0
        if (!keepColor)
            this.colorMap.clear()
        this.setPlayTime(0);
        this.totalTimeMs = 0;
        this.playTimeMs = 0
        this.lastUpdateTime = 0
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

                let x = (t/this.timePreBeat)*this.config.beatWidth
                let y = (127 - pitch)*this.config.noteHeight
                let w = (duration/this.timePreBeat)*this.config.beatWidth
                let h = this.config.noteHeight
                this.svgWidth = Math.ceil(Math.max(x + w, this.svgWidth))
                let color = this.getColor(track, channel)
                let opacity = Math.min(1, velocity/127 + 0.1).toFixed(2)
                let rect = this.drawNote(x,y,w,h, `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`)
                midiEvent.push(rect)
                this.setPlayTime(t);
                this.wrapper.scrollTo(this.svgWidth - this.wrapper.offsetWidth, 0)
            }
            this.midiEvents.push(midiEvent);
            this.svg.style.width = `${this.svgWidth}px`;
        }

    }

    getColor(track, channel){
        let key = `${track},${channel}`;
        let color = this.colorMap.get(key);
        if(!!color){
            return color;
        }
        color = HSVtoRGB(Math.random(),Math.random()*0.5 + 0.5,1);
        this.colorMap.set(key, color);
        return color;
    }

    drawNote(x, y, w, h, fill) {
        if (!this.svg) {
          return null;
        }
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.classList.add('note');
        rect.setAttribute('fill', fill);
        // Round values to the nearest integer to avoid partially filled pixels.
        rect.setAttribute('x', `${Math.round(x)}`);
        rect.setAttribute('y', `${Math.round(y)}`);
        rect.setAttribute('width', `${Math.round(w)}`);
        rect.setAttribute('height', `${Math.round(h)}`);
        this.svg.appendChild(rect);
        return rect
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
    }

    setPlayTime(t){
        this.playTime = t
        let x = Math.round((t/this.timePreBeat)*this.config.beatWidth)
        this.timeLine.setAttribute('x1', `${x}`);
        this.timeLine.setAttribute('y1', '0');
        this.timeLine.setAttribute('x2', `${x}`);
        this.timeLine.setAttribute('y2', `${this.config.noteHeight*128}`);

        this.wrapper.scrollTo(Math.max(0, x - this.wrapper.offsetWidth/2), 0)
        let dt = Date.now() - this.lastUpdateTime; // limit the update rate of ActiveNotes
        if(this.playing && dt > 50){
            let activeNotes = []
            this.removeActiveNotes(this.activeNotes)
            this.midiEvents.forEach((midiEvent)=>{
                if(midiEvent[0] === "note"){
                    let time = midiEvent[1]
                    let duration = midiEvent[3]
                    let note = midiEvent[midiEvent.length - 1]
                    if(time <=this.playTime && time+duration>= this.playTime){
                        activeNotes.push(note)
                    }
                }
            })
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
    let midi_visualizer_container_inited = null
    let midi_audio_audio_inited = null;
    let midi_audio_cursor_inited = null;
    let midi_visualizer = document.createElement('midi-visualizer')
    onUiUpdate((m)=>{
        let app = gradioApp()
        let midi_visualizer_container = app.querySelector("#midi_visualizer_container");
        if(!!midi_visualizer_container && midi_visualizer_container_inited!== midi_visualizer_container){
            midi_visualizer_container.appendChild(midi_visualizer)
            midi_visualizer_container_inited = midi_visualizer_container;
        }
        let midi_audio = app.querySelector("#midi_audio");
        if (!!midi_audio){
            let midi_audio_cursor = midi_audio.deepQuerySelector(".cursor");
            if(!!midi_audio_cursor && midi_audio_cursor_inited!==midi_audio_cursor){
                midi_visualizer.bindWaveformCursor(midi_audio_cursor)
                midi_audio_cursor_inited = midi_audio_cursor
            }
            let midi_audio_audio = midi_audio.deepQuerySelector("audio");
            if(!!midi_audio_audio && midi_audio_audio_inited!==midi_audio_audio){
                midi_visualizer.bindAudioPlayer(midi_audio_audio)
                midi_audio_audio_inited = midi_audio_audio
            }
        }
    })

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
    }

    function removeProgressBar(progressbarContainer){
        let parentProgressbar = progressbarContainer.parentNode;
        let divProgress = parentProgressbar.querySelector(".progressDiv");
        parentProgressbar.removeChild(divProgress);
    }

    function setProgressBar(progressbarContainer, progress, total){
        let parentProgressbar = progressbarContainer.parentNode;
        let divProgress = parentProgressbar.querySelector(".progressDiv");
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
        switch (msg.name) {
            case "visualizer_clear":
                midi_visualizer.clearMidiEvents(false);
                midi_visualizer.version = msg.data
                createProgressBar(midi_visualizer_container_inited)
                break;
            case "visualizer_continue":
                createProgressBar(midi_visualizer_container_inited)
                break;
            case "visualizer_append":
                msg.data.forEach( value => {
                    midi_visualizer.appendMidiEvent(value);
                })
                break;
            case "progress":
                let progress = msg.data[0]
                let total = msg.data[1]
                setProgressBar(midi_visualizer_container_inited, progress, total)
                break;
            case "visualizer_end":
                midi_visualizer.clearMidiEvents(true);
                msg.data.forEach( value => {
                    midi_visualizer.appendMidiEvent(value);
                })
                midi_visualizer.finishAppendMidiEvent()
                midi_visualizer.setPlayTime(0);
                removeProgressBar(midi_visualizer_container_inited);
                break;
            default:
        }
    }
})();
