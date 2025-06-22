import { useEffect, useState } from "react";
import { Modal, Button } from "react-bootstrap";
import adviceList from "../../data/adviceList";

const AIAdviceModal = ({ show, onHide }) => {
  const [randomAdvice, setRandomAdvice] = useState([]);

  useEffect(() => {
    if (show) {
      // Pick 3–5 random unique advice tips
      const shuffled = [...adviceList].sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, 4);
      setRandomAdvice(selected);
    }
  }, [show]);

  return (
    <Modal show={show} onHide={onHide} centered>
      <Modal.Header closeButton>
        <Modal.Title>AI Financial Advice</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {randomAdvice.map((tip, idx) => (
          <p key={idx}>✅ {tip}</p>
        ))}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default AIAdviceModal;
