import React, { useState } from "react";
import { Container, Row, Col, Form, Button, Alert } from "react-bootstrap";
import "../styles/Pages.css";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [formStatus, setFormStatus] = useState({
    submitted: false,
    error: false,
    message: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validation
    if (!formData.name || !formData.email || !formData.message) {
      setFormStatus({
        submitted: true,
        error: true,
        message: "Please fill in all required fields.",
      });
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setFormStatus({
        submitted: true,
        error: true,
        message: "Please enter a valid email address.",
      });
      return;
    }

    // Mock successful form submission
    setFormStatus({
      submitted: true,
      error: false,
      message:
        "Your message has been sent successfully! We will get back to you soon.",
    });

    // Reset form after successful submission
    setFormData({
      name: "",
      email: "",
      subject: "",
      message: "",
    });
  };

  return (
    <div className="contact-page">
      <section className="section hero-section">
        <Container>
          <Row className="align-items-center">
            <Col lg={12} className="text-center">
              <h1 className="section-title">Contact Us</h1>
              <p className="lead">
                Get in touch with our team for any questions or support
              </p>
            </Col>
          </Row>
        </Container>
      </section>

      <section className="section">
        <Container>
          <Row>
            <Col lg={6} className="mb-5 mb-lg-0">
              <h2 className="mb-4">Get In Touch</h2>
              <p>
                Have questions about our services or need support with your
                account? Our team is here to help. Fill out the form and we'll
                get back to you as soon as possible.
              </p>

              <div className="contact-info mt-5">
                <div className="contact-item">
                  <h5>Email</h5>
                  <p>support@financialadvisor.ai</p>
                </div>

                <div className="contact-item">
                  <h5>Phone</h5>
                  <p>+1 (555) 123-4567</p>
                </div>

                <div className="contact-item">
                  <h5>Office Hours</h5>
                  <p>Monday - Friday: 9AM - 6PM EST</p>
                </div>

                <div className="contact-item">
                  <h5>Address</h5>
                  <p>
                    123 Financial Street, Suite 100
                    <br />
                    New York, NY 10001
                  </p>
                </div>
              </div>
            </Col>

            <Col lg={6}>
              <div className="contact-form-container">
                {formStatus.submitted && (
                  <Alert
                    variant={formStatus.error ? "danger" : "success"}
                    className="mb-4"
                  >
                    {formStatus.message}
                  </Alert>
                )}

                <Form onSubmit={handleSubmit}>
                  <Form.Group className="mb-3">
                    <Form.Label>
                      Name <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      placeholder="Your name"
                      required
                    />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>
                      Email <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      placeholder="Your email address"
                      required
                    />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Subject</Form.Label>
                    <Form.Control
                      type="text"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      placeholder="What is this regarding?"
                    />
                  </Form.Group>

                  <Form.Group className="mb-4">
                    <Form.Label>
                      Message <span className="text-danger">*</span>
                    </Form.Label>
                    <Form.Control
                      as="textarea"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      rows={5}
                      placeholder="Your message"
                      required
                    />
                  </Form.Group>

                  <Button type="submit" className="btn btn-primary w-100">
                    Send Message
                  </Button>
                </Form>
              </div>
            </Col>
          </Row>
        </Container>
      </section>

      <section className="section bg-light">
        <Container>
          <Row className="text-center">
            <Col lg={12}>
              <h2 className="mb-5">Frequently Asked Questions</h2>
            </Col>
          </Row>

          <Row>
            <Col lg={6} className="mb-4">
              <div className="faq-item">
                <h4>How does the AI advisor work?</h4>
                <p>
                  Our AI advisor analyzes your financial data, spending
                  patterns, and market trends to provide personalized
                  recommendations tailored to your financial goals and
                  situation.
                </p>
              </div>
            </Col>

            <Col lg={6} className="mb-4">
              <div className="faq-item">
                <h4>Is my financial data secure?</h4>
                <p>
                  Yes, we use bank-level encryption and security measures to
                  protect your data. We never share your personal information
                  with third parties without your consent.
                </p>
              </div>
            </Col>

            <Col lg={6} className="mb-4">
              <div className="faq-item">
                <h4>How much does the service cost?</h4>
                <p>
                  We offer several subscription plans starting at $9.99/month.
                  You can try our service free for 14 days to see if it's right
                  for you.
                </p>
              </div>
            </Col>

            <Col lg={6} className="mb-4">
              <div className="faq-item">
                <h4>Can I cancel my subscription anytime?</h4>
                <p>
                  Absolutely! You can cancel your subscription at any time from
                  your account settings, no questions asked.
                </p>
              </div>
            </Col>
          </Row>
        </Container>
      </section>
    </div>
  );
};

export default Contact;
