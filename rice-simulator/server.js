const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Main route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Endpoint to receive rice grain data
app.post('/api/rice', (req, res) => {
    const riceData = req.body;
    console.log('Rice grain data received:', riceData);
    res.json({ success: true, data: riceData });
});

app.listen(PORT, () => {
    console.log(`🌾 Server running at http://localhost:${PORT}`);
});
