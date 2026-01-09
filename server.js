const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static('public'));

// Proxy endpoint to forward classify requests server-side (avoids CORS)
const axios = require('axios');
const DEFAULT_REMOTE_API = process.env.API_URL || 'http://apirice.endikens.com/';

app.post('/proxy/classify', async (req, res) => {
    const remoteBase = (req.body && req.body._remote_api) ? req.body._remote_api : DEFAULT_REMOTE_API;
    // Remove internal helper prop before forwarding
    const payload = Object.assign({}, req.body);
    if (payload._remote_api) delete payload._remote_api;

    const target = remoteBase.replace(/\/+$/, '') + '/classify/';
    try {
        const resp = await axios.post(target, payload, { timeout: 10000 });
        return res.status(resp.status).json(resp.data);
    } catch (err) {
        console.error('Proxy error forwarding to', target, err.message || err);
        // Attempt to surface useful error to client
        if (err.response) {
            return res.status(err.response.status).json({ error: err.response.data || err.message });
        }
        return res.status(502).json({ error: 'Bad gateway', detail: err.message });
    }
});

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
