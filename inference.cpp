/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "model.h"
#include "inference.h"
#include <Arduino.h>


#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define NUM_CLASSES 6
#define VOTE_WINDOW 5  // sliding window size for majority-vote smoothing


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 32 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

extern float Features_Buffer[40][50];

// The name of this function is important for Arduino compatibility.
void setupModel() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(audio_model_v4_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)


  // Model in python for reference (audio_classifier_v4)
    // model = Sequential([
    //     Conv1D(64, 3, activation='relu', padding='same', input_shape=(40, 50)),
    //     BatchNormalization(),
    //     MaxPooling1D(2),
    //     Conv1D(128, 3, activation='relu', padding='same'),
    //     BatchNormalization(),
    //     MaxPooling1D(2),
    //     Conv1D(128, 3, activation='relu', padding='same'),
    //     BatchNormalization(),
    //     GlobalAveragePooling1D(),
    //     Dense(64, activation='relu'),
    //     Dropout(0.4),   // removed at inference
    //     Dense(32, activation='relu'),
    //     Dropout(0.3),   // removed at inference
    //     Dense(6, activation='softmax')
    // ])
    // Classes (sorted): cat, dog, door, liquid, music, speech

  // BatchNormalization is typically fused into Conv by TFLite converter.
  // If not fused, it decomposes into Mul + Add — both registered below.
  static tflite::MicroMutableOpResolver<12> micro_op_resolver;
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddMean() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddRelu() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddExpandDims() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddDequantize() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddMul() != kTfLiteOk) {
    return;
  }

  if (micro_op_resolver.AddAdd() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  Serial.print("dims->size: "); Serial.println(model_input->dims->size);
  for (int i = 0; i < model_input->dims->size; i++) {
    Serial.print("dim "); Serial.print(i); Serial.print(": ");
    Serial.println(model_input->dims->data[i]);
  }
  Serial.print("type: "); Serial.println(model_input->type);
  if ((model_input->dims->size != 3) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 40) || (model_input->dims->data[2] != 50) ||
      (model_input->type != kTfLiteInt8)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }


  delay(500);
  Serial.print("input_scale: "); Serial.println(model_input->params.scale, 8);
  Serial.print("input_zero_point: "); Serial.println(model_input->params.zero_point);
  delay(500);

  error_reporter->Report("Initialization complete");
}

void runInference() {

    if (interpreter == nullptr || model_input == nullptr) {
        Serial.println("Model not initialized!");
        return;
    }

    float input_scale = model_input->params.scale;
    int input_zero_point = model_input->params.zero_point;
    for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 50; j++) {
            int32_t q = (int32_t)roundf(Features_Buffer[i][j] / input_scale) + input_zero_point;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            model_input->data.int8[i*50 + j] = (int8_t)q;
        }
    }

    TfLiteStatus invoke = interpreter->Invoke();

    if (invoke != kTfLiteOk) {
        error_reporter->Report("Invoke() failed");
        return;
    }

    TfLiteTensor* output = interpreter->output(0);

    // Sorted alphabetically — must match le.classes_ order from training
    const char* CLASS_NAMES[NUM_CLASSES] = {
        "cat", "dog", "door", "liquid", "music", "speech"
    };

    // Per-class confidence thresholds; liquid raised for HVAC environment
    const float CLASS_THRESHOLDS[NUM_CLASSES] = {
        0.50f, // cat
        0.50f, // dog
        0.50f, // door
        0.72f, // liquid
        0.50f, // music
        0.50f, // speech
    };

    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    // Dequantize all outputs and print
    float probs[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = (output->data.int8[i] - output_zero_point) * output_scale;
        Serial.print(CLASS_NAMES[i]);
        Serial.print(": ");
        Serial.print(probs[i], 3);
        Serial.print("  ");
    }
    Serial.println();

    // Pick highest probability that clears its class threshold; else "unknown"
    float max_val = -1e9f;
    int outputclass = -1;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (probs[i] >= CLASS_THRESHOLDS[i] && probs[i] > max_val) {
            max_val = probs[i];
            outputclass = i;
        }
    }
    // Majority-vote smoothing over a sliding window of VOTE_WINDOW frames.
    // Circular buffer stores recent predictions (-1 = "unknown", 0..5 = class).
    // Output is suppressed until the window is full, then the plurality wins.
    // Ties (two or more classes with equal top votes) resolve to "unknown".
    static int vote_buf[VOTE_WINDOW];
    static int vote_head  = 0;
    static int vote_valid = 0;  // saturates at VOTE_WINDOW

    vote_buf[vote_head] = outputclass;
    vote_head = (vote_head + 1) % VOTE_WINDOW;
    if (vote_valid < VOTE_WINDOW) vote_valid++;

    // Wait until window is fully populated
    if (vote_valid < VOTE_WINDOW) return;

    // Tally: indices 0..NUM_CLASSES-1 = named classes, NUM_CLASSES = unknown
    int tally[NUM_CLASSES + 1] = {};
    for (int i = 0; i < VOTE_WINDOW; i++) {
        int cls = vote_buf[i];
        tally[(cls >= 0) ? cls : NUM_CLASSES]++;
    }

    // Find the maximum vote count
    int max_votes = 0;
    for (int i = 0; i <= NUM_CLASSES; i++) {
        if (tally[i] > max_votes) max_votes = tally[i];
    }

    // Detect ties
    int n_winners = 0, winner = -1;
    for (int i = 0; i <= NUM_CLASSES; i++) {
        if (tally[i] == max_votes) { n_winners++; winner = i; }
    }

    Serial.print(">> ");
    if (n_winners > 1) {
        Serial.println("unknown (tied)");
    } else if (winner == NUM_CLASSES) {
        Serial.println("unknown");
    } else {
        Serial.println(CLASS_NAMES[winner]);
    }
}