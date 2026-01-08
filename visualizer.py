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
        p.setup = () => {
            p.createCanvas(400, 300, p.WEBGL);
            p.angleMode(p.DEGREES);
        };

        p.draw = () => {
            p.background(240, 248, 255); // AliceBlue
            
            p.ambientLight(150);
            p.directionalLight(255, 255, 255, 0.5, 0.5, -1);
            p.orbitControl();
            
            p.push();
            p.rotateY(p.frameCount * 0.5);
            p.rotateZ(params.rotation);
            
            let rX = params.minor * 0.5; 
            let rY = params.minor * 0.4; 
            let rZ = params.major * 0.5; 
            
            p.noStroke();
            p.fill(245, 240, 230);
            p.specularMaterial(200);
            p.shininess(20);
            
            p.scale(rZ, rY, rX); 
            p.ellipsoid(1, 1, 1);
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
