import React from 'react';

export const ImageUploader = ({ onUpload, disabled }) => (
  <input
    type="file"
    accept="image/*"
    onChange={(e) => onUpload(e.target.files[0])}
    disabled={disabled}
  />
);