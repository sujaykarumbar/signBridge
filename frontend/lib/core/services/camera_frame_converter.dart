import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts the camera's actual buffer layout, not the requested format.
class CameraFrameConverter {
  static Uint8List toJpeg(
    CameraImage frame, {
    int rotationDegrees = 0,
    int quality = 65,
    int targetSize = 224,
  }) {
    if (frame.width <= 0 || frame.height <= 0) {
      throw const FormatException('Invalid camera frame dimensions');
    }

    switch (frame.format.group) {
      case ImageFormatGroup.yuv420:
        return _fastYuv420ToJpeg(
          frame,
          rotationDegrees: rotationDegrees,
          size: targetSize,
          quality: quality,
        );
      case ImageFormatGroup.bgra8888:
        return _fastBgraToJpeg(
          frame,
          rotationDegrees: rotationDegrees,
          size: targetSize,
          quality: quality,
        );
      case ImageFormatGroup.jpeg:
        return _jpegToJpeg(
          frame,
          rotationDegrees: rotationDegrees,
          size: targetSize,
          quality: quality,
        );
      default:
        final rgb = toRgb(frame);
        var processed = rgb;
        if (rotationDegrees == 90) {
          processed = img.copyRotate(processed, angle: 90);
        } else if (rotationDegrees == 180) {
          processed = img.copyRotate(processed, angle: 180);
        } else if (rotationDegrees == 270) {
          processed = img.copyRotate(processed, angle: 270);
        }
        processed = img.copyResizeCropSquare(processed, size: targetSize);
        return Uint8List.fromList(img.encodeJpg(processed, quality: quality));
    }
  }

  static Uint8List _jpegToJpeg(
    CameraImage frame, {
    int rotationDegrees = 0,
    int size = 224,
    int quality = 65,
  }) {
    if (frame.planes.length != 1) {
      throw const FormatException('Expected one JPEG plane');
    }
    img.Image? decoded;
    try {
      decoded = img.decodeJpg(frame.planes.single.bytes);
    } on img.ImageException {
      throw const FormatException('Invalid JPEG camera frame');
    }
    if (decoded == null) {
      throw const FormatException('Invalid JPEG camera frame');
    }
    var processed = decoded;
    if (rotationDegrees == 90) {
      processed = img.copyRotate(processed, angle: 90);
    } else if (rotationDegrees == 180) {
      processed = img.copyRotate(processed, angle: 180);
    } else if (rotationDegrees == 270) {
      processed = img.copyRotate(processed, angle: 270);
    }
    processed = img.copyResizeCropSquare(processed, size: size);
    return Uint8List.fromList(img.encodeJpg(processed, quality: quality));
  }

  static Uint8List _fastYuv420ToJpeg(
    CameraImage frame, {
    required int rotationDegrees,
    required int size,
    required int quality,
  }) {
    if (frame.planes.length != 3) {
      throw const FormatException('Expected three YUV420 planes');
    }
    final yPlane = frame.planes[0];
    final uPlane = frame.planes[1];
    final vPlane = frame.planes[2];

    final yStride = yPlane.bytesPerPixel ?? 1;
    final uStride = uPlane.bytesPerPixel ?? 1;
    final vStride = vPlane.bytesPerPixel ?? 1;
    final chromaWidth = (frame.width + 1) ~/ 2;
    final chromaHeight = (frame.height + 1) ~/ 2;

    _validatePlane(yPlane, frame.width, frame.height, yStride);
    _validatePlane(uPlane, chromaWidth, chromaHeight, uStride);
    _validatePlane(vPlane, chromaWidth, chromaHeight, vStride);

    final int w = frame.width;
    final int h = frame.height;
    final Uint8List yBytes = yPlane.bytes;
    final Uint8List uBytes = uPlane.bytes;
    final Uint8List vBytes = vPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;
    final int uRowStride = uPlane.bytesPerRow;
    final int vRowStride = vPlane.bytesPerRow;

    final bool isRotated90or270 = (rotationDegrees == 90 || rotationDegrees == 270);
    final int rotW = isRotated90or270 ? h : w;
    final int rotH = isRotated90or270 ? w : h;

    final int cropSide = rotW < rotH ? rotW : rotH;
    final int cropX0 = (rotW - cropSide) ~/ 2;
    final int cropY0 = (rotH - cropSide) ~/ 2;

    final outImage = img.Image(width: size, height: size);

    for (int ty = 0; ty < size; ty++) {
      final int ry = cropY0 + ((ty * cropSide) ~/ size);
      for (int tx = 0; tx < size; tx++) {
        final int rx = cropX0 + ((tx * cropSide) ~/ size);

        int x, y;
        if (rotationDegrees == 90) {
          x = ry;
          y = h - 1 - rx;
        } else if (rotationDegrees == 180) {
          x = w - 1 - rx;
          y = h - 1 - ry;
        } else if (rotationDegrees == 270) {
          x = w - 1 - ry;
          y = rx;
        } else {
          x = rx;
          y = ry;
        }

        if (x < 0) {
          x = 0;
        } else if (x >= w) {
          x = w - 1;
        }
        if (y < 0) {
          y = 0;
        } else if (y >= h) {
          y = h - 1;
        }

        final int yp = yBytes[y * yRowStride + x * yStride];
        final int uvX = x >> 1;
        final int uvY = y >> 1;
        final int up = uBytes[uvY * uRowStride + uvX * uStride];
        final int vp = vBytes[uvY * vRowStride + uvX * vStride];

        final int c = yp;
        final int d = up - 128;
        final int e = vp - 128;

        int r = c + ((1436 * e) >> 10);
        int g = c - ((352 * d + 731 * e) >> 10);
        int b = c + ((1815 * d) >> 10);

        if (r < 0) {
          r = 0;
        } else if (r > 255) {
          r = 255;
        }
        if (g < 0) {
          g = 0;
        } else if (g > 255) {
          g = 255;
        }
        if (b < 0) {
          b = 0;
        } else if (b > 255) {
          b = 255;
        }

        outImage.setPixelRgb(tx, ty, r, g, b);
      }
    }

    return Uint8List.fromList(img.encodeJpg(outImage, quality: quality));
  }

  static Uint8List _fastBgraToJpeg(
    CameraImage frame, {
    required int rotationDegrees,
    required int size,
    required int quality,
  }) {
    if (frame.planes.length != 1) {
      throw const FormatException('Expected one BGRA plane');
    }
    final plane = frame.planes.single;
    final stride = plane.bytesPerPixel ?? 4;
    _validatePlane(plane, frame.width, frame.height, stride, channels: 4);

    final int w = frame.width;
    final int h = frame.height;
    final Uint8List bytes = plane.bytes;
    final int rowStride = plane.bytesPerRow;

    final bool isRotated90or270 = (rotationDegrees == 90 || rotationDegrees == 270);
    final int rotW = isRotated90or270 ? h : w;
    final int rotH = isRotated90or270 ? w : h;

    final int cropSide = rotW < rotH ? rotW : rotH;
    final int cropX0 = (rotW - cropSide) ~/ 2;
    final int cropY0 = (rotH - cropSide) ~/ 2;

    final outImage = img.Image(width: size, height: size);

    for (int ty = 0; ty < size; ty++) {
      final int ry = cropY0 + ((ty * cropSide) ~/ size);
      for (int tx = 0; tx < size; tx++) {
        final int rx = cropX0 + ((tx * cropSide) ~/ size);

        int x, y;
        if (rotationDegrees == 90) {
          x = ry;
          y = h - 1 - rx;
        } else if (rotationDegrees == 180) {
          x = w - 1 - rx;
          y = h - 1 - ry;
        } else if (rotationDegrees == 270) {
          x = w - 1 - ry;
          y = rx;
        } else {
          x = rx;
          y = ry;
        }

        if (x < 0) {
          x = 0;
        } else if (x >= w) {
          x = w - 1;
        }
        if (y < 0) {
          y = 0;
        } else if (y >= h) {
          y = h - 1;
        }

        final int offset = y * rowStride + x * stride;
        outImage.setPixelRgb(
          tx,
          ty,
          bytes[offset + 2],
          bytes[offset + 1],
          bytes[offset],
        );
      }
    }

    return Uint8List.fromList(img.encodeJpg(outImage, quality: quality));
  }

  static img.Image toRgb(CameraImage frame) {
    if (frame.width <= 0 || frame.height <= 0) {
      throw const FormatException('Invalid camera frame dimensions');
    }
    switch (frame.format.group) {
      case ImageFormatGroup.jpeg:
        if (frame.planes.length != 1) {
          throw const FormatException('Expected one JPEG plane');
        }
        img.Image? decoded;
        try {
          decoded = img.decodeJpg(frame.planes.single.bytes);
        } on img.ImageException {
          throw const FormatException('Invalid JPEG camera frame');
        }
        if (decoded == null) {
          throw const FormatException('Invalid JPEG camera frame');
        }
        return decoded;
      case ImageFormatGroup.yuv420:
        return _yuv420(frame);
      case ImageFormatGroup.bgra8888:
        return _bgra(frame);
      default:
        throw FormatException(
            'Unsupported camera format: ${frame.format.group}');
    }
  }

  static void _validatePlane(Plane plane, int width, int height, int stride,
      {int channels = 1}) {
    final rowBytes = (width - 1) * stride + channels;
    final requiredBytes = (height - 1) * plane.bytesPerRow + rowBytes;
    if (stride < channels ||
        plane.bytesPerRow < rowBytes ||
        plane.bytes.length < requiredBytes) {
      throw const FormatException(
          'Invalid camera plane stride or buffer length');
    }
  }

  static img.Image _yuv420(CameraImage frame) {
    if (frame.planes.length != 3) {
      throw const FormatException('Expected three YUV420 planes');
    }
    final y = frame.planes[0];
    final u = frame.planes[1];
    final v = frame.planes[2];
    final yStride = y.bytesPerPixel ?? 1;
    final uStride = u.bytesPerPixel ?? 1;
    final vStride = v.bytesPerPixel ?? 1;
    final chromaWidth = (frame.width + 1) ~/ 2;
    final chromaHeight = (frame.height + 1) ~/ 2;
    _validatePlane(y, frame.width, frame.height, yStride);
    _validatePlane(u, chromaWidth, chromaHeight, uStride);
    _validatePlane(v, chromaWidth, chromaHeight, vStride);
    final rgb = img.Image(width: frame.width, height: frame.height);
    for (var row = 0; row < frame.height; row++) {
      for (var col = 0; col < frame.width; col++) {
        final yp = y.bytes[row * y.bytesPerRow + col * yStride];
        final up = u.bytes[(row ~/ 2) * u.bytesPerRow + (col ~/ 2) * uStride];
        final vp = v.bytes[(row ~/ 2) * v.bytesPerRow + (col ~/ 2) * vStride];
        rgb.setPixelRgb(
          col,
          row,
          (yp + 1.402 * (vp - 128)).round().clamp(0, 255),
          (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
              .round()
              .clamp(0, 255),
          (yp + 1.772 * (up - 128)).round().clamp(0, 255),
        );
      }
    }
    return rgb;
  }

  static img.Image _bgra(CameraImage frame) {
    if (frame.planes.length != 1) {
      throw const FormatException('Expected one BGRA plane');
    }
    final plane = frame.planes.single;
    final stride = plane.bytesPerPixel ?? 4;
    _validatePlane(plane, frame.width, frame.height, stride, channels: 4);
    final rgb = img.Image(width: frame.width, height: frame.height);
    for (var row = 0; row < frame.height; row++) {
      for (var col = 0; col < frame.width; col++) {
        final offset = row * plane.bytesPerRow + col * stride;
        // Index the Uint8List view itself so its byte offset is respected.
        rgb.setPixelRgb(col, row, plane.bytes[offset + 2],
            plane.bytes[offset + 1], plane.bytes[offset]);
      }
    }
    return rgb;
  }
}
