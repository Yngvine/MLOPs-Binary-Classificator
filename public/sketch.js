// Background music
let bgAudio = null;

// Rice grain variables
let riceParams = {
    majorAxisLength: 120,
    minorAxisLength: 40,
    eccentricity: 0.8,
    roundness: 0.6,
    aspectRatio: 3
};

let autoRotate = true;
let rotationY = 0;

function setup() {
    let canvas = createCanvas(800, 600, WEBGL);
    canvas.parent('p5Canvas');

    // Setup form controls
    setupControls();

    // Calculate initial values
    updateCalculatedValues();

    // Setup background music with native JavaScript Audio
    setupBackgroundMusic();
}

function draw() {
    // Sky gradient background
    background(135, 206, 235);

    // Low poly lighting with shadows
    ambientLight(100);
    directionalLight(255, 255, 255, 0.5, 0.5, -1);

    push();

    // Automatic rotation
    if (autoRotate) {
        rotationY += 0.02;
    }

    // Allow rotation with mouse
    if (mouseIsPressed) {
        rotationY = map(mouseX, 0, width, -PI, PI);
    }

    rotateY(rotationY);
    rotateX(0.2);

    // Draw rice grain
    drawRiceGrain();

    pop();

    // Draw ground plane
    drawGround();
}

function drawRiceGrain() {
    push();

    // Rice grain color (creamy white)
    fill(245, 240, 230);
    noStroke();

    // Apply material with low poly style
    ambientMaterial(245, 240, 230);
    specularMaterial(255, 250, 240);
    shininess(30);

    // Scale according to parameters
    let scaleX = riceParams.minorAxisLength * 0.5;
    let scaleY = riceParams.minorAxisLength * 0.5 * riceParams.roundness;
    let scaleZ = riceParams.majorAxisLength * 0.5;

    scale(scaleX, scaleY, scaleZ);

    // Create grain shape (low poly ellipsoid)
    drawEllipsoid();

    pop();
}

function drawEllipsoid() {
    // Low poly ellipsoid
    sphere(1, 8, 6);
}

function drawGround() {
    push();

    // Position ground below
    translate(0, 150, 0);
    rotateX(HALF_PI);

    // Green grass material
    fill(100, 200, 100);
    noStroke();
    ambientMaterial(100, 200, 100);

    // Draw low poly ground plane
    plane(800, 800, 6, 6);

    pop();
}

function drawAxes() {
    // X Axis - Red
    push();
    stroke(255, 0, 0);
    strokeWeight(2);
    line(0, 0, 0, 100, 0, 0);
    pop();

    // Y Axis - Green
    push();
    stroke(0, 255, 0);
    strokeWeight(2);
    line(0, 0, 0, 0, 100, 0);
    pop();

    // Z Axis - Blue
    push();
    stroke(0, 0, 255);
    strokeWeight(2);
    line(0, 0, 0, 0, 0, 100);
    pop();
}

function setupControls() {
    // Major Axis
    const majorAxis = document.getElementById('majorAxis');
    const majorAxisValue = document.getElementById('majorAxisValue');
    majorAxis.addEventListener('input', (e) => {
        riceParams.majorAxisLength = parseFloat(e.target.value);
        majorAxisValue.textContent = e.target.value;
        updateCalculatedValues();
    });

    // Minor Axis
    const minorAxis = document.getElementById('minorAxis');
    const minorAxisValue = document.getElementById('minorAxisValue');
    minorAxis.addEventListener('input', (e) => {
        riceParams.minorAxisLength = parseFloat(e.target.value);
        minorAxisValue.textContent = e.target.value;
        updateCalculatedValues();
    });

    // Eccentricity (optional control)
    const eccentricity = document.getElementById('eccentricity');
    const eccentricityValue = document.getElementById('eccentricityValue');
    if (eccentricity && eccentricityValue) {
        eccentricity.addEventListener('input', (e) => {
            riceParams.eccentricity = parseFloat(e.target.value);
            eccentricityValue.textContent = e.target.value;
            updateCalculatedValues();
        });
    }

    // Roundness (optional control)
    const roundness = document.getElementById('roundness');
    const roundnessValue = document.getElementById('roundnessValue');
    if (roundness && roundnessValue) {
        roundness.addEventListener('input', (e) => {
            riceParams.roundness = parseFloat(e.target.value);
            roundnessValue.textContent = e.target.value;
            updateCalculatedValues();
        });
    }

    // Aspect Ratio (optional control)
    const aspectRatio = document.getElementById('aspectRatio');
    const aspectRatioValue = document.getElementById('aspectRatioValue');
    if (aspectRatio && aspectRatioValue) {
        aspectRatio.addEventListener('input', (e) => {
            riceParams.aspectRatio = parseFloat(e.target.value);
            aspectRatioValue.textContent = e.target.value;
            updateCalculatedValues();
        });
    }

    // Reset button
    document.getElementById('resetBtn').addEventListener('click', () => {
        resetToDefaults();
    });

    // Random button
    document.getElementById('randomBtn').addEventListener('click', () => {
        randomizeParameters();
    });

    // Toggle auto-rotate on canvas click
    const canvas = document.getElementById('p5Canvas');
    if (canvas) {
        canvas.addEventListener('dblclick', () => {
        autoRotate = !autoRotate;
        });
    }
}

function updateCalculatedValues() {
    // Calculate approximate ellipse area
    const a = riceParams.majorAxisLength / 2;
    const b = riceParams.minorAxisLength / 2;
    const area = Math.PI * a * b;
    document.getElementById('areaValue').textContent = area.toFixed(2);

    // Calculate approximate perimeter (Ramanujan's formula)
    const h = Math.pow((a - b), 2) / Math.pow((a + b), 2);
    const perimeter = Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h)));
    document.getElementById('perimeterValue').textContent = perimeter.toFixed(2);

    // Equivalent diameter
    const equivDiameter = 2 * Math.sqrt(area / Math.PI);
    document.getElementById('equivDiameterValue').textContent = equivDiameter.toFixed(2);

    // Extent (area ratio)
    const extent = riceParams.roundness * 0.85; // Approximation
    const extEl = document.getElementById('extentValue');
    if (extEl) extEl.textContent = extent.toFixed(3);
}

function resetToDefaults() {
    riceParams = {
        majorAxisLength: 120,
        minorAxisLength: 40,
        eccentricity: 0.8,
        roundness: 0.6,
        aspectRatio: 3
    };

    document.getElementById('majorAxis').value = 120;
    document.getElementById('majorAxisValue').textContent = '120';
    document.getElementById('minorAxis').value = 40;
    document.getElementById('minorAxisValue').textContent = '40';
    const eccEl = document.getElementById('eccentricity');
    const eccVal = document.getElementById('eccentricityValue');
    if (eccEl && eccVal) { eccEl.value = 0.8; eccVal.textContent = '0.8'; }
    const rndEl = document.getElementById('roundness');
    const rndVal = document.getElementById('roundnessValue');
    if (rndEl && rndVal) { rndEl.value = 0.6; rndVal.textContent = '0.6'; }
    const arEl = document.getElementById('aspectRatio');
    const arVal = document.getElementById('aspectRatioValue');
    if (arEl && arVal) { arEl.value = 3; arVal.textContent = '3'; }

    updateCalculatedValues();
}

function randomizeParameters() {
    const randomMajor = Math.floor(Math.random() * 150) + 50;
    const randomMinor = Math.floor(Math.random() * 80) + 20;
    const randomEcc = (Math.random() * 0.8 + 0.2).toFixed(2);
    const randomRound = (Math.random() * 0.6 + 0.4).toFixed(2);
    const randomAspect = (Math.random() * 4 + 1).toFixed(1);

    riceParams.majorAxisLength = randomMajor;
    riceParams.minorAxisLength = randomMinor;
    riceParams.eccentricity = parseFloat(randomEcc);
    riceParams.roundness = parseFloat(randomRound);
    riceParams.aspectRatio = parseFloat(randomAspect);

    document.getElementById('majorAxis').value = randomMajor;
    document.getElementById('majorAxisValue').textContent = randomMajor;
    document.getElementById('minorAxis').value = randomMinor;
    document.getElementById('minorAxisValue').textContent = randomMinor;
    const eccEl2 = document.getElementById('eccentricity');
    const eccVal2 = document.getElementById('eccentricityValue');
    if (eccEl2 && eccVal2) { eccEl2.value = randomEcc; eccVal2.textContent = randomEcc; }
    const rndEl2 = document.getElementById('roundness');
    const rndVal2 = document.getElementById('roundnessValue');
    if (rndEl2 && rndVal2) { rndEl2.value = randomRound; rndVal2.textContent = randomRound; }
    const arEl2 = document.getElementById('aspectRatio');
    const arVal2 = document.getElementById('aspectRatioValue');
    if (arEl2 && arVal2) { arEl2.value = randomAspect; arVal2.textContent = randomAspect; }

    updateCalculatedValues();
}

function mouseWheel(event) {
    // Zoom with mouse wheel
    return false; // Prevent page scroll
}

function setupBackgroundMusic() {
    bgAudio = new Audio('background-music.mp3');
    bgAudio.loop = true;
    bgAudio.volume = 0.5;

    // Play on first user interaction
    document.addEventListener('click', function playAudio() {
        bgAudio.play().catch(e => console.log('Audio play failed:', e));
        document.removeEventListener('click', playAudio);
    }, { once: true });

    // Setup mute button
    const muteBtn = document.getElementById('muteBtn');
    muteBtn.addEventListener('click', () => {
        if (bgAudio.muted) {
            bgAudio.muted = false;
            muteBtn.textContent = '🔊';
        } else {
            bgAudio.muted = true;
            muteBtn.textContent = '🔇';
        }
    });
}
