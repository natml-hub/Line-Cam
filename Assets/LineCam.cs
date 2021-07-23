/* 
*   Line Cam
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using NatSuite.Devices;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;
    using NatSuite.ML.Visualizers;

    public class LineCam : MonoBehaviour {
        
        [Header(@"NatML")]
        public string accessKey;

        [Header(@"Visualization")]
        public LineSegmentVisualizer visualizer;

        CameraDevice cameraDevice;
        Texture2D previewTexture;
        MLModelData modelData;
        MLModel model;
        LineSegmentPredictor predictor;

        async void Start () {
            // Request camera permissions
            if (!await MediaDeviceQuery.RequestPermissions<CameraDevice>()) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            previewTexture = await cameraDevice.StartRunning();
            // Display the camera preview
            visualizer.Render(previewTexture);
            // Fetch the model data
            Debug.Log("Fetching model from NatML Hub");
            modelData = await MLModelData.FromHub("@natsuite/line-segment-detector", accessKey);
            // Deserialize the model
            model = modelData.Deserialize();
            // Create the line segment predictor
            predictor = new LineSegmentPredictor(model);
        }

        void Update () {
            // Check that the model has been downloaded
            if (predictor == null)
                return;
            // Create input feature
            var input = new MLImageFeature(previewTexture);
            (input.mean, input.std) = modelData.normalization;
            // Predict
            var lines = predictor.Predict(input);
            // Visualize
            visualizer.Render(previewTexture, lines);
        }

        void OnDisable () {
            // Dispose the model
            model?.Dispose();
            // Stop the camera preview
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
        }
    }
}