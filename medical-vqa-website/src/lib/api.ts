const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export type ModelType = 'hybrid' | 'traditional';

export type PredictionResponse = {
  answer: string;
  organ: string;
  modality: string;
  confidence: number;
  risk: string;
  evidence: string;
  debug?: {
    filename?: string;
    image_sha256?: string;
    question_intent?: string;
    model_type?: string;
  };
  error?: string;
  detail?: string;
};

export async function predictMedicalImage(
  file: File,
  question: string,
  modelType: ModelType,
): Promise<PredictionResponse> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 90000);
  const formData = new FormData();
  formData.append('file', file);
  formData.append('question', question);
  formData.append('modelType', modelType);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    const data = (await response.json()) as PredictionResponse;
    if (!response.ok) {
      throw new Error(data.error ?? data.detail ?? 'Prediction request failed.');
    }

    return data;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('The backend took too long to respond. Restart the API server and try again.');
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}
