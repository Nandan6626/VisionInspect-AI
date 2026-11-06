if __name__ == "__main__":
    from ultralytics import YOLO

    # Load your trained classification model
    model = YOLO("models/best.pt")

    # Test image
    image_path = "test_images/component.jpg"

    # Run classification
    results = model.predict(source=image_path, save=True)
    # Display predictions
    for result in results:

        print("\n===== VISIONINSPECT AI =====")

        probabilities = result.probs

        if probabilities is not None:
            top_indices = probabilities.top5

            for index in top_indices:
                class_name = model.names[index]
                confidence = float(probabilities.data[index]) * 100

                print(f"{class_name}: {confidence:.2f}%")

        else:
            print("No classification probabilities available.")

    print("\nInspection completed!")
