import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:provider/provider.dart';

import '../../../core/services/api_service.dart';
import '../../auth/providers/auth_provider.dart';
import '../providers/translation_provider.dart';

class VideoUploadScreen extends StatefulWidget {
  const VideoUploadScreen({super.key});

  @override
  State<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends State<VideoUploadScreen> {
  final ImagePicker _picker = ImagePicker();
  final FlutterTts _tts = FlutterTts();

  XFile? _selectedVideo;
  bool _isProcessing = false;
  String _processingStep = '';
  Map<String, dynamic>? _translationResult;
  String? _errorMessage;
  bool _isSpeaking = false;
  bool _isSaved = false;

  static const primaryColor = Color(0xFF2B2D5D);
  static const accentColor = Color(0xFF4B6CF7);
  static const successColor = Color(0xFF2BB673);
  static const warningColor = Color(0xFFB7791F);
  static const bgColor = Color(0xFFF5F7FF);

  @override
  void initState() {
    super.initState();
    _initTts();
  }

  void _initTts() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    _tts.setStartHandler(() {
      if (mounted) setState(() => _isSpeaking = true);
    });
    _tts.setCompletionHandler(() {
      if (mounted) setState(() => _isSpeaking = false);
    });
  }

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }

  Future<void> _pickVideo(ImageSource source) async {
    try {
      final file = await _picker.pickVideo(
        source: source,
        maxDuration: const Duration(minutes: 3),
      );
      if (file == null) return;

      setState(() {
        _selectedVideo = file;
        _translationResult = null;
        _errorMessage = null;
        _isSaved = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to pick video: $e';
      });
    }
  }

  Future<void> _translateVideo() async {
    if (_selectedVideo == null) return;

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
      _processingStep = 'Analyzing hand movements and ISL signs...';
    });

    try {
      final res = await ApiService.translateVideo(_selectedVideo!.path);
      if (!mounted) return;

      final sentence = res['translated_sentence']?.toString() ??
          res['sentence']?.toString() ??
          '';
      final confidence = (res['confidence'] as num?)?.toDouble() ?? 0;
      final requiresReview = res['requires_review'] == true;

      if (sentence.isNotEmpty && confidence >= 0.45 && !requiresReview) {
        context.read<TranslationProvider>().updateTranslation(sentence);
      }

      setState(() {
        _isProcessing = false;
        _translationResult = res;
        _processingStep = '';
      });

      if (sentence.isNotEmpty && confidence >= 0.45 && !requiresReview) {
        await _tts.speak(sentence);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Video translation error: $e';
        _processingStep = '';
      });
    }
  }

  Future<void> _saveTranslation() async {
    if (_translationResult == null || _isSaved) return;

    final sentence = _translationResult!['translated_sentence']?.toString() ??
        _translationResult!['sentence']?.toString() ??
        '';
    if (sentence.isEmpty) return;

    final auth = context.read<AuthProvider>();
    final translationProvider = context.read<TranslationProvider>();

    try {
      await translationProvider.saveTranslation(
        auth.userId ?? 'dev_user',
        sentence,
        token: auth.token,
      );
      if (mounted) {
        setState(() => _isSaved = true);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Translation saved to history!'),
            backgroundColor: successColor,
            duration: Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        backgroundColor: primaryColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded,
              color: Colors.white, size: 20),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Upload ISL Video',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header Info Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF2B2D5D), Color(0xFF4B6CF7)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: accentColor.withValues(alpha: 0.25),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.15),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.video_library_rounded,
                        color: Colors.white, size: 26),
                  ),
                  const SizedBox(width: 14),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Continuous ISL Video Translation',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 15,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          'Upload reels, social clips, or recorded signs for AI translation.',
                          style: TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Video Picker Area
            if (_selectedVideo == null) ...[
              GestureDetector(
                onTap: () => _pickVideo(ImageSource.gallery),
                child: Container(
                  height: 220,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                        color: accentColor.withValues(alpha: 0.35), width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: accentColor.withValues(alpha: 0.1),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.cloud_upload_rounded,
                          size: 44,
                          color: accentColor,
                        ),
                      ),
                      const SizedBox(height: 14),
                      const Text(
                        'Select Video from Gallery',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: primaryColor,
                        ),
                      ),
                      const SizedBox(height: 6),
                      const Text(
                        'MP4, MOV, continuous ISL video clips',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.black45,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 14),

              // Camera Record Button Option
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  side: const BorderSide(color: accentColor),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
                onPressed: () => _pickVideo(ImageSource.camera),
                icon: const Icon(Icons.videocam_rounded, color: accentColor),
                label: const Text(
                  'Record New Video with Camera',
                  style: TextStyle(
                      color: accentColor, fontWeight: FontWeight.bold),
                ),
              ),
            ] else ...[
              // Selected Video Card
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.05),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: const Color(0xFFE6FAF3),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.check_circle_rounded,
                              color: successColor, size: 24),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _selectedVideo!.name,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 14,
                                  color: primaryColor,
                                ),
                              ),
                              const SizedBox(height: 2),
                              const Text(
                                'Ready for translation',
                                style: TextStyle(
                                    fontSize: 11, color: Colors.black45),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.change_circle_outlined,
                              color: accentColor),
                          tooltip: 'Change Video',
                          onPressed: _isProcessing
                              ? null
                              : () => _pickVideo(ImageSource.gallery),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    if (!_isProcessing && _translationResult == null)
                      ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: accentColor,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          elevation: 2,
                        ),
                        onPressed: _translateVideo,
                        icon: const Icon(Icons.translate_rounded),
                        label: const Text(
                          'Translate Video',
                          style: TextStyle(
                              fontSize: 15, fontWeight: FontWeight.bold),
                        ),
                      ),
                  ],
                ),
              ),
            ],

            // Processing Indicator
            if (_isProcessing) ...[
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.05),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    const CircularProgressIndicator(color: accentColor),
                    const SizedBox(height: 16),
                    Text(
                      _processingStep,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 13,
                        color: primaryColor,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'Tracking both hands and translating the signed phrase...',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 11, color: Colors.black45),
                    ),
                  ],
                ),
              ),
            ],

            // Error Display
            if (_errorMessage != null) ...[
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.error_outline_rounded,
                        color: Colors.red.shade700),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _errorMessage!,
                        style:
                            TextStyle(color: Colors.red.shade800, fontSize: 13),
                      ),
                    ),
                  ],
                ),
              ),
            ],

            // Translation Result Display
            if (_translationResult != null) ...[
              const SizedBox(height: 24),
              _buildResultCard(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final res = _translationResult!;
    final sentence = res['translated_sentence']?.toString() ??
        res['sentence']?.toString() ??
        '';
    final glosses =
        (res['detected_glosses'] as List?)?.map((e) => e.toString()).toList() ??
            [];
    final conf = ((res['confidence'] as num?)?.toDouble() ?? 0.0) * 100;
    final signType = res['sign_type']?.toString() ?? 'continuous_sentence';
    final requiresReview = res['requires_review'] == true;
    final isConfident = conf >= 45 && !requiresReview;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 14,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: (isConfident ? successColor : warningColor)
                      .withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isConfident
                          ? Icons.verified_rounded
                          : Icons.warning_amber_rounded,
                      color: isConfident ? successColor : warningColor,
                      size: 14,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      requiresReview
                          ? '${conf.toStringAsFixed(0)}% AI estimate - verify'
                          : '${conf.toStringAsFixed(0)}% ${isConfident ? 'Confidence' : 'Match - verify'}',
                      style: TextStyle(
                        color: isConfident ? successColor : warningColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: accentColor.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  signType.replaceAll('_', ' ').toUpperCase(),
                  style: const TextStyle(
                    color: accentColor,
                    fontWeight: FontWeight.w600,
                    fontSize: 10,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          const Text(
            'ENGLISH TRANSLATION',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              letterSpacing: 1.1,
              color: Colors.black45,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            sentence.isNotEmpty ? sentence : 'No clear gestures detected.',
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: primaryColor,
              height: 1.3,
            ),
          ),

          if ((res['message']?.toString() ?? '').isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(
              res['message'].toString(),
              style: const TextStyle(
                color: warningColor,
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ],
          if (requiresReview) ...[
            const SizedBox(height: 10),
            const Text(
              'AI-estimated from visible hand signs only. Verify the sentence before relying on it.',
              style: TextStyle(
                color: warningColor,
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ],

          if (glosses.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text(
              'DETECTED ISL GLOSSES',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.1,
                color: Colors.black45,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: glosses.map((g) {
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEEF2FF),
                    borderRadius: BorderRadius.circular(8),
                    border:
                        Border.all(color: accentColor.withValues(alpha: 0.3)),
                  ),
                  child: Text(
                    g,
                    style: const TextStyle(
                      color: accentColor,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],

          if ((_translationResult?['hand_gestures_observed']?.toString() ?? '')
              .isNotEmpty) ...[
            const SizedBox(height: 14),
            const Text(
              'PHYSICAL HAND SIGNS OBSERVED',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.1,
                color: Colors.black45,
              ),
            ),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.black12),
              ),
              child: Text(
                _translationResult!['hand_gestures_observed'].toString(),
                style: const TextStyle(
                  fontSize: 12,
                  color: Color(0xFF334155),
                  height: 1.4,
                ),
              ),
            ),
          ],

          const SizedBox(height: 20),
          const Divider(),
          const SizedBox(height: 12),

          // Action Buttons: Speak, Save, Reset
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFE67E22),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10)),
                  ),
                  onPressed: () => _tts.speak(sentence),
                  icon: Icon(
                      _isSpeaking
                          ? Icons.volume_up_rounded
                          : Icons.volume_up_outlined,
                      size: 18),
                  label: Text(_isSpeaking ? 'Speaking...' : 'Speak'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isSaved ? Colors.grey : successColor,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10)),
                  ),
                  onPressed: _isSaved ? null : _saveTranslation,
                  icon: Icon(_isSaved ? Icons.check : Icons.save_alt_rounded,
                      size: 18),
                  label: Text(_isSaved ? 'Saved' : 'Save'),
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {
              setState(() {
                _selectedVideo = null;
                _translationResult = null;
                _errorMessage = null;
                _isSaved = false;
              });
            },
            icon: const Icon(Icons.refresh_rounded, size: 18),
            label: const Text('Upload Another Video'),
          ),
        ],
      ),
    );
  }
}
