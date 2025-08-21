// ChatBox.tsx (수정본)
import { useState, useEffect, useRef } from "react";
import type { ChangeEvent } from "react";
import {
  Box,
  Button,
  Paper,
  TextField,
  Typography,
  CircularProgress,
  Collapse,
  Stack,
  Divider,
} from "@mui/material";
import { generate } from "../api";
import type { DocumentChunk, GenerateRequest } from "../types";
import VoiceInput from "./VoiceInput";

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m > 0 ? `${m}분 ` : ""}${s}초`;
}

export default function ChatBox() {
  const [question, setQuestion] = useState("");
  const [prompt, setPrompt] = useState("");
  const [docs, setDocs] = useState<DocumentChunk[] | null>([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false); // ⬅️ 추가

  const [elapsed, setElapsed] = useState(0);
  const [finalElapsed, setFinalElapsed] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const latestTranscriptRef = useRef<string>("");

  const handleAsk = async (forcedQuestion?: string) => {
    const q = (forcedQuestion ?? question).trim();
    if (!q) return;

    setAnswer("");
    setPrompt("");
    setDocs([]);

    setLoading(true);
    setElapsed(0);
    setFinalElapsed(null);
    startTimeRef.current = Date.now();

    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    try {
      const req: GenerateRequest = { query: q }; // ← 여기만 꼭 q로!
      const res = await generate(req);
      setDocs(
        Array.isArray(res.reference_documents) ? res.reference_documents : []
      );
      setPrompt(res.prompt);
      setAnswer(res.response);
    } finally {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setFinalElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100vw",
        bgcolor: "#f3f4f6",
        py: 6,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
      }}
    >
      <Paper
        elevation={4}
        sx={{
          width: "100%",
          maxWidth: 800,
          p: 4,
          borderRadius: 3,
          bgcolor: "#ffffff",
        }}
      >
        <Typography variant="h5" gutterBottom>
          🧠 RAG 챗봇
        </Typography>

        {/* 🔊 음성 입력: 전사는 question에 바로 반영 */}
        <VoiceInput
          // 2) 전사 수신 시 상태와 ref 둘 다 갱신
          onTranscriptChange={(text) => {
            setQuestion(text);
            latestTranscriptRef.current = text;
          }}
          // 3) 종료 시 ref 값으로 즉시 호출 → 상태 지연 이슈 제거
          onListeningChange={(isListening) => {
            setListening(isListening);
            if (!isListening) {
              const finalText = latestTranscriptRef.current.trim();
              if (finalText) handleAsk(finalText);
            }
          }}
        />

        {/* 키보드 보조 입력(그대로 유지) */}
        <TextField
          label="질문을 입력하세요"
          variant="outlined"
          fullWidth
          value={question}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            setQuestion(e.target.value)
          }
          sx={{ mt: 2, mb: 2 }}
          disabled={listening} // 듣는 동안 잠시 비활성화(선택)
        />
        <Box textAlign="center" mt={4}>
          <Button
            variant="contained"
            onClick={() => handleAsk()}
            disabled={loading}
          >
            {loading ? `답변 생성 중... (${elapsed}초)` : "질문하기"}
          </Button>
        </Box>

        {/* 질문 영역 */}

        <Divider sx={{ my: 3 }} />

        {/* 로딩 중 UI */}
        {loading && (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              py: 6,
            }}
          >
            <Stack spacing={2} alignItems="center">
              <CircularProgress />
              <Typography color="text.secondary">
                답변 생성 중... {elapsed > 0 ? `(${elapsed}초 경과)` : ""}
              </Typography>
            </Stack>
          </Box>
        )}

        {/* 결과 영역: 로딩이 아닐 때만 등장 */}
        <Collapse in={!loading && !!answer} timeout={300} unmountOnExit>
          <Box
            sx={{
              mt: 2,
              p: 2,
              bgcolor: "#f9fafb",
              borderRadius: 2,
              border: "1px solid #e0e0e0",
            }}
          >
            <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
              답변:
            </Typography>
            <Typography sx={{ whiteSpace: "pre-wrap" }}>{answer}</Typography>

            {finalElapsed !== null && (
              <Typography sx={{ mt: 2, color: "gray" }}>
                ⏱️ 답변 시간: {formatElapsed(finalElapsed)}
              </Typography>
            )}
          </Box>

          {docs && docs.length > 0 && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: "#f9fafb",
                borderRadius: 2,
                border: "1px solid #e0e0e0",
                overflow: "hidden",
              }}
            >
              <Typography
                variant="subtitle1"
                sx={{ fontWeight: "bold", mb: 2 }}
              >
                📚 참고자료
              </Typography>
              {docs.map((doc, index) => (
                <Box key={index} sx={{ mb: 2, overflowWrap: "break-word" }}>
                  <Typography variant="body1" component="div">
                    <a
                      href={doc.url || "#"}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        fontWeight: "bold",
                        textDecoration: "underline",
                        color: "#1976d2",
                        wordBreak: "break-word",
                      }}
                      onClick={(e) => {
                        if (!doc.url) e.preventDefault();
                      }}
                    >
                      {doc.title || "제목 없음"}
                    </a>
                  </Typography>
                  {doc.chunk_text && (
                    <Typography
                      variant="body2"
                      sx={{
                        mt: 0.5,
                        color: "text.secondary",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {doc.chunk_text}
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          )}

          {prompt && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: "#f9fafb",
                borderRadius: 2,
                border: "1px solid #e0e0e0",
              }}
            >
              <Typography
                variant="subtitle1"
                sx={{ fontWeight: "bold", mb: 1 }}
              >
                Prompt:
              </Typography>
              <Typography sx={{ whiteSpace: "pre-wrap" }}>{prompt}</Typography>
            </Box>
          )}
        </Collapse>
      </Paper>
    </Box>
  );
}
