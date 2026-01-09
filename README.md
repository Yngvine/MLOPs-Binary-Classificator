# RicePainter 🌾

Interactive web application for visualizing rice grains in 3D with Node.js, Express and p5.js.

## Features

- ✨ 3D rendering of rice grains with low poly style
- 🔄 Automatic 360° rotation
- 🌍 Green ground plane and blue sky
- 🎛️ Interactive controls to adjust parameters
- 📊 Automatic calculation of grain properties

## Installation

1. Install dependencies:

```bash
npm install
```

2. Start the server:

```bash
npm start
```

3. Open in browser:

```
http://localhost:3000
```

## Development Mode

For development with auto-reload:

```bash
npm run dev
```

## Grain Parameters

### Interactive Parameters:

- **MajorAxisLength**: Major axis length
- **MinorAxisLength**: Minor axis length
- **Eccentricity**: Grain eccentricity
- **Roundness**: Grain roundness
- **AspectRatio**: Aspect ratio

### Calculated Values:

- **Area**: Grain area
- **Perimeter**: Perimeter
- **EquivDiameter**: Equivalent diameter
- **Extent**: Extent

## Controls

- 🖱️ **Drag**: Manually rotate the grain
- 💫 **Double click**: Enable/disable automatic rotation
- 🎲 **Random**: Generate random parameters
- ↩️ **Reset**: Return to default values

## Technologies

- Node.js
- Express
- p5.js (WebGL)
- HTML5/CSS3

## License

MIT
