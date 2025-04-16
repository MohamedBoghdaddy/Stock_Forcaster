import { ReactNode } from "react";

export interface User {
  id: string;
  name: string;
  email: string;
  role?: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
}

export type AuthAction =
  | { type: "LOGIN_SUCCESS" | "USER_LOADED"; payload: User }
  | { type: "LOGOUT_SUCCESS" | "AUTH_ERROR" };

export interface AuthContextType {
  state: AuthState;
  dispatch: React.Dispatch<AuthAction>;
  logout: () => void;
}

export interface AuthProviderProps {
  children: ReactNode;
}
