import { useState, useEffect, useRef } from "react";
import type { ChangeEvent } from "react";
import { Box, Button, Paper, TextField, Typography } from "@mui/material";
import { generate } from "../api";
import type { DocumentChunk, GenerateRequest } from "../types";

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

  const [elapsed, setElapsed] = useState(0);
  const [finalElapsed, setFinalElapsed] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setElapsed(0);
    setFinalElapsed(null);
    startTimeRef.current = Date.now();

    intervalRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    const req: GenerateRequest = {
      query: question
    };
    const res = await generate(req);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    const totalSeconds = Math.floor((Date.now() - startTimeRef.current) / 1000);
    setFinalElapsed(totalSeconds);

    setDocs(Array.isArray(res.reference_documents) ? res.reference_documents : []);
    setPrompt(res.prompt);
    setAnswer(res.response);
    setLoading(false);
  };

  // 정리 함수
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

        <TextField
          label="질문을 입력하세요"
          variant="outlined"
          fullWidth
          value={question}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setQuestion(e.target.value)}
          sx={{ mt: 2, mb: 2 }}
        />

        <Button
          variant="contained"
          fullWidth
          onClick={handleAsk}
          disabled={loading}
          sx={{ py: 1.2, fontWeight: "bold" }}
        >
          {loading ? `답변 생성 중... (${elapsed}초)` : "질문하기"}
        </Button>

        {answer && (
          <Box
            sx={{
              mt: 4,
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
        )}

        {(docs && docs.length > 0) && (
          <Box
            sx={{
              mt: 4,
              p: 2,
              bgcolor: "#f9fafb",
              borderRadius: 2,
              border: "1px solid #e0e0e0",
              overflow: "hidden",
            }}
          >
            <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 2 }}>
              📚 참고자료
            </Typography>

            {docs.map((doc, index) => (
              <Box
                key={index}
                sx={{
                  mb: 2,
                  overflowWrap: "break-word", // 긴 텍스트 방지
                }}
              >
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
                      if (!doc.url) {
                        e.preventDefault(); // url 없으면 링크 차단
                      }
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
              mt: 4,
              p: 2,
              bgcolor: "#f9fafb",
              borderRadius: 2,
              border: "1px solid #e0e0e0",
            }}
          >
            <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
              Prompt:
            </Typography>
            <Typography sx={{ whiteSpace: "pre-wrap" }}>{prompt}</Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
