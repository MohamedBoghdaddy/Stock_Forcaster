// server/validators/profileValidator.js
import { body } from "express-validator";

export const profileValidationRules = [
  body("age")
    .optional()
    .isInt({ min: 18, max: 120 })
    .withMessage("Age must be between 18 and 120"),
  body("employmentStatus")
    .optional()
    .isIn(["Employed", "Self-employed", "Unemployed", "Student", "Retired"]),
  body("salary").optional().isNumeric(),
  body("financialGoals").optional().isString().trim().escape(),
];
