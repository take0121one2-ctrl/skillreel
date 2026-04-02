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
    return res.status(400).json({ error: 'APIキーが設定されていません' });
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
        model: 'claude-opus-4-5',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: [
            {
              type: 'text',
              text: `飲食チェーンの品質管理として正解画像と提出画像を比較してください。\nチェックポイント：\n${(checkPoints || []).map((p, i) => `${i + 1}. ${p}`).join('\n')}\n\n以下のJSON形式のみで回答してください（前後に余分なテキスト不要）：\n{"result":"pass","score":85,"comment":"フィードバック内容","details":["項目1のコメント","項目2のコメント"]}`
            },
            { type: 'text', text: '【正解画像】' },
            { type: 'image', source: { type: 'base64', media_type: correctMediaType || 'image/jpeg', data: correctImageBase64 } },
            { type: 'text', text: '【提出画像】' },
            { type: 'image', source: { type: 'base64', media_type: submittedMediaType || 'image/jpeg', data: submittedImageBase64 } },
          ]
        }]
      })
    });

    const responseText = await response.text();

    if (!response.ok) {
      return res.status(500).json({ error: `Anthropic API error: ${response.status} - ${responseText}` });
    }

    const data = JSON.parse(responseText);
    const text = data.content?.[0]?.text || '';
    const cleaned = text.replace(/```json|```/g, '').trim();
    const result = JSON.parse(cleaned);
    return res.status(200).json(result);

  } catch (err) {
    return res.status(500).json({ error: err.message, stack: err.stack });
  }
}
