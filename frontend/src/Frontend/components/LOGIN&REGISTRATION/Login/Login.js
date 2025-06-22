import React from "react";
import "../../styles/login.css"; // Adjust the path as needed
import { Link, useNavigate } from "react-router-dom";
import { useLogin } from "../../../../hooks/useLogin";
import ShowPass from "../../../../assets/eye.svg";
import ShowPassOff from "../../../../assets/eye-off.svg";

const Login = () => {
  const {
    email,
    setEmail,
    password,
    setPassword,
    showPassword,
    setShowPassword,
    errorMessage,
    successMessage,
    isLoading,
    handleLogin,
  } = useLogin();

  return (
    <div className="main-container">
      <div className="login-container">
        <div className="left-login">
          <h2>Login</h2>
          <form style={{ width: "90%", margin: "auto" }} onSubmit={handleLogin}>
            <div className="field">
              <div className="field-wrapper">
                <label className="custom_label" htmlFor="email">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  placeholder="enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>
            <div className="field password-container">
              <div className="field-wrapper">
                <label className="custom_label" htmlFor="password">
                  Password
                </label>
                <div className="password-container">
                  <input
                    placeholder="enter your password"
                    className="password_field"
                    type={showPassword ? "text" : "password"}
                    id="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="show-password"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <img src={ShowPassOff} alt="" />
                    ) : (
                      <img src={ShowPass} alt="" />
                    )}
                  </button>
                </div>
              </div>
            </div>
            {errorMessage && <div className="error">{errorMessage}</div>}
            {successMessage && <div className="success">{successMessage}</div>}
            <button className="left_btn" type="submit" disabled={isLoading}>
              {isLoading ? "Logging in..." : "Login"}
            </button>
          </form>
        </div>

        <div className="right-login">
          <h4 className="reg_cta">Don't have an account?</h4>
          <Link to="/signup" className="reg_link">
            <button className="right_btn" type="button" disabled={isLoading}>
              Signup
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
