export const config = {
  api: {
    bodyParser: {
      sizeLimit: '20mb',
    },
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { correctImageBase64, correctMediaType, submittedImageBase64, submittedMediaType, checkPoints } = req.body;

  const apiKey = req.headers['x-anthropic-key'];
  if (!apiKey) {
    return res.status(400).json({ error: 'API key missing' });
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-opus-4-6',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: [
            {
              type: 'text',
              text: `飲食チェーンの品質管理として正解画像と提出画像を比較してください。\nチェックポイント：\n${checkPoints.map((p, i) => `${i + 1}. ${p}`).join('\n')}\n\n以下のJSON形式のみで回答：\n{"result":"pass","score":85,"comment":"フィードバック","details":["各項目コメント"]}`
            },
            { type: 'text', text: '【正解画像】' },
            { type: 'image', source: { type: 'base64', media_type: correctMediaType, data: correctImageBase64 } },
            { type: 'text', text: '【提出画像】' },
            { type: 'image', source: { type: 'base64', media_type: submittedMediaType, data: submittedImageBase64 } },
          ]
        }]
      })
    });

    const data = await response.json();
    const text = data.content?.[0]?.text || '';
    const result = JSON.parse(text.replace(/```json|```/g, '').trim());
    res.status(200).json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
