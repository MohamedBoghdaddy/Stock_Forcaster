import mongoose from "mongoose";

const UserSchema = new mongoose.Schema(
  {
    username: { type: String, required: true, unique: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    gender: { type: String, required: true },
    firstName: { type: String, required: true },
    lastName: { type: String, required: true },
    role: {
      type: String,
      enum: ["user", "admin"],
      default: "user",
    },
    receiveNotifications: {
      type: Boolean,
      default: true,
    },
    profilePhoto: {
      type: String,
    },

    blocked: {
      type: Boolean,
      default: false,
    },

    blocked: {
      type: Boolean,
      default: false,
    },

    // models/UserModel.js

    lastLogin: {
      type: Date,
    },
    lastIP: {
      type: String,
    },
    activityLog: [
      {
        action: String,
        timestamp: Date,
      },
    ],
    profilePhoto: {
      type: String, // URL or path
    },
  },
  { timestamps: true }
);

const User = mongoose.model("User", UserSchema);

export default User;
