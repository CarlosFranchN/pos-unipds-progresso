importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIM = 640;
const CLASS_TRHRESHOLD = 0.4;
let _labels = []
let _model = null

async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH)

    // warmup

    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)

    tf.dispose(dummyInput)

    postMessage({ type: 'model-loaded' })
}

// pre processa a imagem p o formato aceito pelo YOLO
// tf.browser.fromPixels converte a imagem para um tensor: converte ImageBitmap/ImageData para tensor [H,W,3]
// tf.image.resizeBilinear redimensiona o tensor para o tamanho aceito pelo modelo para [INPUT_DIM , INPUT_DIM]
// .div(255): normaliza os valores dos pixels para o intervalo [0, 1] (dividindo por 255)
// .expandDims(0): adiciona uma dimensão extra no início do tensor para representar o batch size, resultando em um tensor de forma [1, INPUT_DIM, INPUT_DIM, 3]
// tf.tidy()
// - garante q o tensores temporarios serao descartados automaticamente
// evitando vazemnto de memoria

function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input)

        return tf.image
        .resizeBilinear(image, [INPUT_MODEL_DIM, INPUT_MODEL_DIM])
        .div(255)
        .expandDims(0)
    })
}


async function runInference(tensor) {
    const output = await _model.executeAsync(tensor)
    tf.dispose(tensor)
    
    const [boxes, scores, classes] = output.slice(0,3)
    const [boxesData, scoresData, classesData] = await Promise.all(
        [boxes.data(), scores.data(), classes.data()]
    )

    output.forEach(t =>t.dispose())
    
    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}

function* processPredictions({ boxes, scores, classes }, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_TRHRESHOLD) continue

        const label = _labels[classes[i]]
        if (label !== 'kite') continue

        let [x1,y1,x2,y2] = boxes.slice(i*4,(i+1)*4)
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        
        const boxHeight = y2 - y1
        const boxWidth = x2 - x1

        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2

        yield {
            x: centerX,
            y: centerY,
            score: (scores[i] * 100).toFixed(2),
            label
        }
        

    }
}

loadModelAndLabels()
self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return

    const input = preprocessImage(data.image)
    const  {width, height} = data.image
    const inferenceResults = await runInference(input)
    // processPredictions(inferenceResults, width, height)
    for (const pred of processPredictions(inferenceResults, width, height)) {
        console.log(pred);
        postMessage({
            type: 'prediction',
            ...pred
        });
    }
    // postMessage({
    //     type: 'prediction',
    //     x: 400,
    //     y: 400,
    //     score: 0
    // });



};

console.log('🧠 YOLOv5n Web Worker initialized');
