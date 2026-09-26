# Chatterbox Português do Brasil

Worker de serverless da Runpod para o modelo [ResembleAI/Chatterbox-Multilingual-pt-br](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br). A fala sai em português do Brasil. A clonagem usa o áudio enviado no pedido.

Os pesos não vão neste repositório. A imagem baixa o modelo brasileiro na build e só fica pronta se os pesos encaixarem.

## Pedido

`POST https://api.runpod.ai/v2/{endpoint_id}/runsync`

```json
{
  "input": {
    "text": "Oi, tudo bem?",
    "reference_audio_url": "https://exemplo/voz.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8
  }
}
```

`language_id` é sempre `pt`. Sem `reference_audio_url`, a voz padrão também é brasileira.

A resposta traz `audio_base64`, `sample_rate`, `language` (`pt-BR`) e `cloned`.
