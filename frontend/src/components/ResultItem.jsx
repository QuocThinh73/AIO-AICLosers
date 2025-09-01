function ResultItem({ src, name }) {
  return (
    <>
      <img src={src} alt={name} style={{ objectFit: "cover", border: "1px solid #ddd" }} />
      <span>{name}</span>
    </>
  );
}

export default ResultItem;