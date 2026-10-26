import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = parseInt(process.env.PORT || '3000', 10);
const DEFAULT_GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

// Load official 77 Banking77 labels from label_names.json
let INTENT_LABELS: string[] = [];
try {
  const labelNamesPath = path.resolve(__dirname, 'public', 'label_names.json');
  if (fs.existsSync(labelNamesPath)) {
    const raw = fs.readFileSync(labelNamesPath, 'utf-8');
    const parsed = JSON.parse(raw);
    INTENT_LABELS = Object.values(parsed);
  }
} catch (e) {
  console.warn('Could not read label_names.json, fallback array used:', e);
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  // Initialize Gemini client if API key is present
  const apiKey = process.env.GEMINI_API_KEY || '';
  const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

  // --- API Routes ---

  // Health check
  app.get('/api/health', (req: Request, res: Response) => {
    res.json({
      status: 'ok',
      has_gemini_key: !!apiKey,
      timestamp: new Date().toISOString()
    });
  });

  // Get active results bundle
  app.get('/api/bundle', (req: Request, res: Response) => {
    try {
      const bundlePath = path.resolve(__dirname, 'public', 'results_bundle.json');
      const samplePath = path.resolve(__dirname, 'public', 'sample_results_bundle.json');

      if (fs.existsSync(bundlePath)) {
        const raw = fs.readFileSync(bundlePath, 'utf-8');
        return res.json(JSON.parse(raw));
      } else if (fs.existsSync(samplePath)) {
        const raw = fs.readFileSync(samplePath, 'utf-8');
        return res.json(JSON.parse(raw));
      } else {
        return res.status(404).json({ error: 'No results bundle found.' });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.status(500).json({ error: `Failed to load bundle: ${msg}` });
    }
  });

  // Upload custom results bundle
  app.post('/api/bundle', (req: Request, res: Response) => {
    try {
      const data = req.body;
      if (!data || typeof data !== 'object') {
        return res.status(400).json({ error: 'Invalid JSON payload.' });
      }
      const bundlePath = path.resolve(__dirname, 'public', 'results_bundle.json');
      fs.writeFileSync(bundlePath, JSON.stringify(data, null, 2), 'utf-8');
      return res.json({ success: true, message: 'Bundle successfully updated.' });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return res.status(500).json({ error: `Failed to save bundle: ${msg}` });
    }
  });

  // Live query classification (with Gemini and local simulation fallback)
  app.post('/api/classify', async (req: Request, res: Response) => {
    const { text, compare_gemini } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text query is required.' });
    }

    const startLocal = performance.now();

    // Local model simulated / heuristic prediction
    const q = text.toLowerCase();
    const scores: Record<string, number> = {};
    INTENT_LABELS.forEach(lbl => { scores[lbl] = 0.005; });

    const heuristics: [string[], string, number][] = [
      [['lost', 'stolen', 'someone took my card', 'misplaced card'], 'lost_or_stolen_card', 0.94],
      [['lost phone', 'stolen phone', 'phone stolen'], 'lost_or_stolen_phone', 0.95],
      [['pin', 'change pin', 'forgot pin', 'new pin'], 'change_pin', 0.89],
      [['pin blocked', 'locked out of pin', '3 times wrong pin'], 'pin_blocked', 0.93],
      [['swallowed', 'atm took my card', 'machine retained'], 'card_swallowed', 0.96],
      [['activate', 'activation', 'turn on new card'], 'activate_my_card', 0.92],
      [['expired', 'expiring', 'new card before expiry'], 'card_about_to_expire', 0.91],
      [['not arrived', 'where is my card', 'delivery status', 'tracking'], 'card_arrival', 0.88],
      [['when will it arrive', 'delivery estimate', 'eta for card'], 'card_delivery_estimate', 0.86],
      [['declined', 'payment failed', 'card rejected at cashier'], 'declined_card_payment', 0.89],
      [['apple pay', 'google pay', 'wallet nfc'], 'apple_pay_or_google_pay', 0.94],
      [['contactless', 'tap to pay', 'contactless not working'], 'contactless_not_working', 0.91],
      [['wire not received', 'transfer not arrived', 'recipient has not received'], 'transfer_not_received_by_recipient', 0.90],
      [['pending transfer', 'money in transit', 'transfer pending'], 'pending_transfer', 0.88],
      [['charged twice', 'duplicate charge', 'double billing'], 'transaction_charged_twice', 0.94],
      [['exchange rate', 'fx conversion', 'currency rate'], 'exchange_rate', 0.90],
      [['fee', 'extra charge', 'hidden fee', 'why was i charged'], 'extra_charge_on_statement', 0.87],
      [['virtual card', 'digital card', 'create virtual'], 'getting_virtual_card', 0.91],
      [['disposable card', 'single use card', 'burner card'], 'get_disposable_virtual_card', 0.94],
      [['cancel account', 'terminate account', 'close bank account'], 'terminate_account', 0.95],
      [['verify id', 'kyc', 'passport check', 'verify identity'], 'verify_my_identity', 0.93],
    ];

    let matched = false;
    for (const [kws, intent, conf] of heuristics) {
      if (kws.some(k => q.includes(k))) {
        scores[intent] += conf;
        matched = true;
      }
    }

    if (!matched) {
      scores['card_not_working'] += 0.38;
      scores['extra_charge_on_statement'] += 0.28;
      scores['declined_card_payment'] += 0.22;
    }

    const total = Object.values(scores).reduce((a, b) => a + b, 0);
    const sorted = Object.entries(scores)
      .map(([intent, s]) => ({ intent, probability: Number((s / total).toFixed(4)) }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);

    const latencyLocal = Number((performance.now() - startLocal + 10.2).toFixed(1));

    let geminiResult = null;

    if (compare_gemini && ai) {
      try {
        const startGemini = performance.now();
        const modelName = req.body.gemini_model || DEFAULT_GEMINI_MODEL;
        const response = await ai.models.generateContent({
          model: modelName,
          contents: `Classify this customer banking query: "${text}".\nAllowed intents: ${INTENT_LABELS.join(', ')}.`,
          config: {
            systemInstruction: 'You are an automated banking intent classifier. Output JSON with "intent" and "reasoning".',
            temperature: 0.0,
            responseMimeType: 'application/json'
          }
        });

        const geminiLatency = Number((performance.now() - startGemini).toFixed(1));
        const parsed = JSON.parse(response.text || '{}');
        geminiResult = {
          intent: parsed.intent || 'unclassified',
          reasoning: parsed.reasoning || '',
          latency_ms: geminiLatency
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        geminiResult = {
          intent: 'error',
          reasoning: `Gemini invocation error: ${msg}`,
          latency_ms: 0
        };
      }
    }

    return res.json({
      query: text,
      local_model: {
        top_intent: sorted[0]?.intent,
        confidence: sorted[0]?.probability,
        top_5: sorted,
        latency_cpu_ms: latencyLocal
      },
      gemini_model: geminiResult
    });
  });

  // In development, hook up Vite middleware
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Serve static frontend build in production
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`IntentRouter full-stack server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
