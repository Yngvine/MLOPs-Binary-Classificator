import html

def get_rice_visualizer_html():
    html_content = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
  <style>
    body { margin: 0; overflow: hidden; display: flex; justify-content: center; align-items: center; background: transparent; }
    canvas { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  </style>
</head>
<body>
  <script>
    let params = { major: 164.16, minor: 50.17, rotation: 0 };
    
    window.addEventListener('message', (event) => {
         const data = event.data;
         if (data.type === 'update_params') {
             params.major = parseFloat(data.major);
             params.minor = parseFloat(data.minor);
             params.rotation = parseFloat(data.rotation);
         }
    });

    const s = (p) => {
        let camAngleY = 0;
        
        // Icosahedron Data
        let icoVertices = [];
        let icoIndices = [];

        p.setup = () => {
            p.createCanvas(400, 300, p.WEBGL);
            p.angleMode(p.DEGREES);
            
            // Define Unit Icosahedron
            const t = (1.0 + Math.sqrt(5.0)) / 2.0;
            const rawVerts = [
                [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
                [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
                [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
            ];
            
            icoVertices = rawVerts.map(v => {
                 let m = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                 return p.createVector(v[0]/m, v[1]/m, v[2]/m);
            });
            
            icoIndices = [
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ];

            // Subdivide Logic (Function)
            function subdivide() {
                let nextIndices = [];
                let midCache = {}; 
                
                function getMidPoint(i1, i2) {
                    let key = i1 < i2 ? i1 + "_" + i2 : i2 + "_" + i1;
                    if (key in midCache) return midCache[key];
                    
                    let v1 = icoVertices[i1];
                    let v2 = icoVertices[i2];
                    let mid = p5.Vector.add(v1, v2).div(2).normalize();
                    
                    icoVertices.push(mid);
                    let index = icoVertices.length - 1;
                    midCache[key] = index;
                    return index;
                }

                for (let face of icoIndices) {
                    let v0 = face[0];
                    let v1 = face[1];
                    let v2 = face[2];
                    
                    let a = getMidPoint(v0, v1);
                    let b = getMidPoint(v1, v2);
                    let c = getMidPoint(v2, v0);
                    
                    nextIndices.push([v0, a, c]);
                    nextIndices.push([v1, b, a]);
                    nextIndices.push([v2, c, b]);
                    nextIndices.push([a, b, c]);
                }
                icoIndices = nextIndices;
            }

            // Apply Subdivision twice (Level 2 Icosphere) for sufficient geometry
            subdivide();
            subdivide();
            
            // --- Apply "Germ Notch" (Asymmetric dent at one tip) ---
            // Target: +Z tip (Extreme), +X/+Y Quadrant (Side)
            // We define a center point for the "missing chunk" on the surface of the unit sphere
            let notchCenter = p.createVector(0.8, 0.0, 0.3); 
            let notchRadius = 0.9;

            for (let v of icoVertices) {
                let d = p5.Vector.dist(v, notchCenter);
                
                if (d < notchRadius) {
                    // Create a smooth scoop shape
                    // 1.0 at center, 0.0 at edge
                    let falloff = Math.cos((d / notchRadius) * (Math.PI / 2));
                    
                    // Dig deeper in the middle
                    let indent = 0.35 * falloff;
                    
                    // Apply reduction to vertex radius
                    v.mult(1.0 - indent);
                }
            }
        };

        p.draw = () => {
            // Dark "Lab" Background
            p.background(30, 35, 40); 
            
            p.ambientLight(60);
            p.directionalLight(255, 255, 255, 0.8, 0.8, -0.5);
            p.pointLight(100, 100, 255, -200, -200, 200);

            // Manual Interaction: Restrict to Horizontal (Rotates around Y-axis)
            if (p.mouseIsPressed) {
                camAngleY -= (p.mouseX - p.pmouseX) * 0.5;
            }
            // Auto-rotate slowly logic
            else {
                 camAngleY += 0.2;
            }
            
            p.push();
            
            // 1. View Transformations
            // Tilt slightly (X-axis) to see 3D volume. Negative angle looks from "above" (p5 coords).
            p.rotateX(-25); 
            // Apply Drag Rotation (Y-axis)
            p.rotateY(camAngleY);
            
            // --- DRAW GRID FLOOR (Fixed relative to grain rotation) ---
            p.push();
            p.translate(0, 80, 0); // Move "floor" down
            p.rotateX(90);
            p.stroke(60);
            p.strokeWeight(1);
            p.noFill();
            let gSize = 600;
            let step = 50;
            for(let i = -gSize/2; i <= gSize/2; i+=step){
                p.line(i, -gSize/2, i, gSize/2);
                p.line(-gSize/2, i, gSize/2, i);
            }
            p.pop();
            // -----------------------------------------------------------
            
            // 2. Object Transformations (from Sliders)
            // Apply Grain Rotation (Z-axis)
            p.rotateZ(params.rotation);
            
            let rX = params.minor * 0.5; 
            let rY = params.minor * 0.4; 
            let rZ = params.major * 0.5; 
            
            p.noStroke();
            p.fill(245, 240, 230);
            p.specularMaterial(255);
            p.shininess(10);
            
            p.scale(rZ, rY, rX); 
            
            // Icosahedron Draw (Faceted)
            p.beginShape(p.TRIANGLES);
            for (let face of icoIndices) {
                let v1 = icoVertices[face[0]];
                let v2 = icoVertices[face[1]];
                let v3 = icoVertices[face[2]];
                
                // Calculate Flat Normal
                let u = p5.Vector.sub(v2, v1);
                let v = p5.Vector.sub(v3, v1);
                let n = u.cross(v).normalize();
                
                p.normal(n.x, n.y, n.z);
                p.vertex(v1.x, v1.y, v1.z);
                p.vertex(v2.x, v2.y, v2.z);
                p.vertex(v3.x, v3.y, v3.z);
            }
            p.endShape();

            p.pop();
        };
    };
    new p5(s);
  </script>
</body>
</html>
"""
    
    escaped_src = html.escape(html_content)
    
    return f'''
    <iframe id="rice-visualizer-iframe" 
            style="width: 100%; height: 320px; border: none; border-radius: 8px;" 
            srcdoc="{escaped_src}">
    </iframe>
    '''
