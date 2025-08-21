import axios from 'axios';
import type { AxiosResponse } from 'axios';
import type { GenerateResponse, GenerateRequest } from './types';

const API_URL = import.meta.env.VITE_API_URL;

export async function generate(req: GenerateRequest): Promise<GenerateResponse> {
  const res: AxiosResponse<GenerateResponse> = await axios.post(`${API_URL}/generate`, req);
  return res.data;
}
