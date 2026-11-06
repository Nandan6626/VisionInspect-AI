import os
from django import forms
from django.core.exceptions import ValidationError


ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label="Industrial Component Photograph",
        help_text="Upload a clear photograph of the manufactured part (JPG, JPEG, PNG, up to 20MB)",
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'id': 'componentImageInput',
            'accept': 'image/jpeg,image/png,image/jpg,image/webp'
        })
    )

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise ValidationError("Please provide a valid component image file.")

        ext = os.path.splitext(image.name)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )

        if image.size > MAX_IMAGE_FILE_SIZE:
            size_mb = image.size / (1024 * 1024)
            raise ValidationError(
                f"Uploaded file size ({size_mb:.1f} MB) exceeds maximum allowed size (20 MB)."
            )

        return image

