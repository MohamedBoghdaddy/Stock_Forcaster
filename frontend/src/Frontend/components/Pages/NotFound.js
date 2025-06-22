import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import "../styles/Pages.css";

const NotFound = () => {
  return (
    <div className="not-found-page">
      <Container className="text-center">
        <Row className="justify-content-center">
          <Col md={8} lg={6}>
            <div className="error-container">
              <h1 className="error-code">404</h1>
              <h2 className="error-title">Page Not Found</h2>
              <p className="error-message">
                The page you are looking for might have been removed, had its
                name changed, or is temporarily unavailable.
              </p>
              <Link to="/">
                <Button variant="primary" className="mt-4">
                  Return to Home
                </Button>
              </Link>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default NotFound;
