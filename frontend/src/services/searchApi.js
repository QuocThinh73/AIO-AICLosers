const baseSearch = async ({
  query,
  ocr,
  image,
  translate,
  includeBatchIds,
  excludeBatchIds,
  topK,
}) => {
  const fd = new FormData();
  fd.append("use_translation", translate ? "true" : "false");
  if (query?.trim()) fd.append("embedding_text", query.trim());
  if (ocr?.trim()) fd.append("ocr_text", ocr.trim());
  if (image) fd.append("embedding_image", image);
  fd.append("top_k", String(Number(topK) || 100));

  if (includeBatchIds?.trim())
    fd.append("include_batch_ids", includeBatchIds.trim());
  if (excludeBatchIds?.trim())
    fd.append("exclude_batch_ids", excludeBatchIds.trim());

  const res = await fetch("/api/search/base_search", {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
};

export default baseSearch;
