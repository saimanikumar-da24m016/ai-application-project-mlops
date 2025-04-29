import axios from "axios";
const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const uploadImage = (file) => {
  const data = new FormData();
  data.append("file", file);
  return axios.post(`${BASE}/predict`, data, {
    headers: { "Content-Type": "multipart/form-data" }
  });
};

export const getDriftStatus = () =>
  axios.get(`${BASE}/drift-status`);

export const sendFeedback = (feedback) =>
  axios.post(`${BASE}/feedback`, feedback);
