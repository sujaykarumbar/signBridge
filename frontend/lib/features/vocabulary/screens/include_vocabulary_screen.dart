import 'package:flutter/material.dart';
import '../../../core/services/api_service.dart';
import '../../translate/screens/translate_screen.dart';

class IncludeVocabularyScreen extends StatefulWidget {
  const IncludeVocabularyScreen({super.key});

  @override
  State<IncludeVocabularyScreen> createState() => _IncludeVocabularyScreenState();
}

class _IncludeVocabularyScreenState extends State<IncludeVocabularyScreen> {
  bool _isLoading = true;
  String? _errorMessage;

  List<String> _categories = ['All'];
  String _selectedCategory = 'All';
  bool _coreOnly = false;
  String _searchQuery = '';

  List<Map<String, dynamic>> _allSigns = [];
  Map<String, dynamic>? _datasetInfo;
  final TextEditingController _searchCtrl = TextEditingController();

  static const Color primaryColor = Color(0xFF1E2158);
  static const Color accentColor = Color(0xFF4B6CF7);
  static const Color pageBg = Color(0xFFF5F7FF);

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final infoFuture = ApiService.fetchIncludeInfo().catchError((_) => <String, dynamic>{});
      final catsFuture = ApiService.fetchIncludeCategories().catchError((_) => ['All']);
      final signsFuture = ApiService.fetchIncludeSigns(coreOnly: false).catchError((_) => <Map<String, dynamic>>[]);

      final results = await Future.wait([infoFuture, catsFuture, signsFuture]);
      final info = results[0] as Map<String, dynamic>;
      final cats = results[1] as List<String>;
      final signs = results[2] as List<Map<String, dynamic>>;

      if (mounted) {
        setState(() {
          _datasetInfo = info.isNotEmpty ? info : null;
          _categories = cats.isNotEmpty ? cats : ['All'];
          _allSigns = signs;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Could not load INCLUDE signs from backend: $e';
          _isLoading = false;
        });
      }
    }
  }

  List<Map<String, dynamic>> get _filteredSigns {
    return _allSigns.where((s) {
      final label = (s['label'] ?? '').toString().toLowerCase();
      final id = (s['id'] ?? '').toString().toLowerCase();
      final cat = (s['category'] ?? '').toString();
      final isCore = s['is_core_50'] == true;

      if (_coreOnly && !isCore) return false;
      if (_selectedCategory != 'All' &&
          cat.toLowerCase() != _selectedCategory.toLowerCase()) {
        return false;
      }
      if (_searchQuery.isNotEmpty) {
        final q = _searchQuery.toLowerCase();
        if (!label.contains(q) && !id.contains(q) && !cat.toLowerCase().contains(q)) {
          return false;
        }
      }
      return true;
    }).toList();
  }

  void _showDatasetInfoSheet() {
    final totalVideos = _datasetInfo?['total_videos'] != null
        ? '${_datasetInfo!['total_videos']} Videos'
        : '4,292 Videos';
    final catsCount = _datasetInfo?['categories_count'] != null
        ? '${_datasetInfo!['categories_count']} Semantic Categories'
        : '15 Semantic Categories';
    final totalClasses = _datasetInfo?['total_classes'] != null
        ? '${_datasetInfo!['total_classes']} Classes'
        : '263 Indian Sign Language Classes';
    final weights = _datasetInfo?['trained_model_weights']?.toString() ??
        'include50_no_cnn_lstm.pth (85.6 MB)';

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.black12,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 18),
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEEF2FF),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.dataset_outlined,
                      color: accentColor, size: 24),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'INCLUDE Dataset',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: primaryColor,
                        ),
                      ),
                      Text(
                        'Zenodo Record 4010759',
                        style: TextStyle(fontSize: 12, color: Colors.black54),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text(
              'A Large Scale Dataset for Indian Sign Language Recognition (ACM Multimedia 2020) by AI4Bharat & IIT Madras.',
              style: TextStyle(fontSize: 13, height: 1.5, color: Colors.black87),
            ),
            const SizedBox(height: 16),
            _infoRow(Icons.movie_outlined, 'Total Clips', totalVideos),
            _infoRow(Icons.category_outlined, 'Categories', catsCount),
            _infoRow(Icons.translate_outlined, 'Vocabulary', totalClasses),
            _infoRow(Icons.psychology_outlined, 'Model Weights', weights),
            const SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: accentColor,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Close', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoRow(IconData icon, String title, String val) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Icon(icon, size: 16, color: accentColor),
          const SizedBox(width: 10),
          Text(title, style: const TextStyle(fontSize: 13, color: Colors.black54)),
          const Spacer(),
          Text(val,
              style: const TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w600, color: primaryColor)),
        ],
      ),
    );
  }

  void _showSignDetails(Map<String, dynamic> sign) {
    final String label = sign['label']?.toString() ?? 'Sign';
    final String cat = sign['category']?.toString() ?? 'General';
    final bool isCore = sign['is_core_50'] == true;

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.black12,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                CircleAvatar(
                  radius: 26,
                  backgroundColor: isCore
                      ? const Color(0xFFE6FAF3)
                      : const Color(0xFFEEF2FF),
                  child: Text(
                    label.isNotEmpty ? label[0].toUpperCase() : '?',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: isCore ? const Color(0xFF0F6E56) : accentColor,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        label,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: primaryColor,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.05),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              cat,
                              style: const TextStyle(
                                  fontSize: 11, color: Colors.black54),
                            ),
                          ),
                          if (isCore) ...[
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 2),
                              decoration: BoxDecoration(
                                color: const Color(0xFFE6FAF3),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: const Text(
                                'Core 50 Class',
                                style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold,
                                    color: Color(0xFF0F6E56)),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Dataset Source',
                    style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.black45),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Zenodo Record 4010759 • AI4Bharat INCLUDE ISL',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        color: primaryColor),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    isCore
                        ? 'High-speed local neural recognition via AI4Bharat BiLSTM model with real-time pose extraction.'
                        : 'Recognized via hybrid live pipeline with Gemini Multimodal ISL Stream.',
                    style: const TextStyle(fontSize: 12, color: Colors.black54, height: 1.4),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: accentColor,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 0,
                    ),
                    onPressed: () {
                      Navigator.pop(ctx);
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const TranslateScreen(),
                        ),
                      );
                    },
                    icon: const Icon(Icons.videocam_rounded, size: 20),
                    label: const Text(
                      'Practice with Camera',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final signs = _filteredSigns;

    return Scaffold(
      backgroundColor: pageBg,
      body: Column(
        children: [
          // Header
          Container(
            decoration: const BoxDecoration(
              color: primaryColor,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(24),
                bottomRight: Radius.circular(24),
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(7),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.menu_book_rounded,
                              color: Colors.white, size: 20),
                        ),
                        const SizedBox(width: 10),
                        const Text(
                          'ISL Dictionary',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 20,
                          ),
                        ),
                        const Spacer(),
                        IconButton(
                          icon: const Icon(Icons.info_outline, color: Colors.white70),
                          onPressed: _showDatasetInfoSheet,
                          tooltip: 'INCLUDE Dataset Info',
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Explore 263 signs from the AI4Bharat INCLUDE dataset (Zenodo 4010759).',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.75),
                        fontSize: 12,
                        height: 1.4,
                      ),
                    ),
                    const SizedBox(height: 14),

                    // Search input
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(14),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.08),
                            blurRadius: 10,
                            offset: const Offset(0, 3),
                          ),
                        ],
                      ),
                      child: TextField(
                        controller: _searchCtrl,
                        onChanged: (val) => setState(() => _searchQuery = val),
                        decoration: InputDecoration(
                          hintText: 'Search sign (e.g. Hello, Doctor, Water)...',
                          hintStyle: const TextStyle(fontSize: 13, color: Colors.black38),
                          prefixIcon: const Icon(Icons.search, color: primaryColor, size: 20),
                          suffixIcon: _searchQuery.isNotEmpty
                              ? IconButton(
                                  icon: const Icon(Icons.clear, size: 18),
                                  onPressed: () {
                                    _searchCtrl.clear();
                                    setState(() => _searchQuery = '');
                                  },
                                )
                              : null,
                          border: InputBorder.none,
                          contentPadding:
                              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Filters & Category chips
          Container(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      '${signs.length} signs found',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: primaryColor,
                      ),
                    ),
                    const Spacer(),
                    FilterChip(
                      selected: _coreOnly,
                      label: const Text('Core 50 Only', style: TextStyle(fontSize: 11)),
                      selectedColor: const Color(0xFFE6FAF3),
                      checkmarkColor: const Color(0xFF0F6E56),
                      labelStyle: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: _coreOnly ? const Color(0xFF0F6E56) : Colors.black87,
                      ),
                      onSelected: (val) => setState(() => _coreOnly = val),
                    ),
                  ],
                ),
                const SizedBox(height: 8),

                // Category chips
                SizedBox(
                  height: 36,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: _categories.length,
                    separatorBuilder: (_, __) => const SizedBox(width: 8),
                    itemBuilder: (context, idx) {
                      final cat = _categories[idx];
                      final isSelected = cat == _selectedCategory;
                      return ChoiceChip(
                        label: Text(cat),
                        selected: isSelected,
                        onSelected: (_) => setState(() => _selectedCategory = cat),
                        selectedColor: accentColor,
                        labelStyle: TextStyle(
                          fontSize: 12,
                          color: isSelected ? Colors.white : primaryColor,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                        ),
                        backgroundColor: Colors.white,
                        side: BorderSide(
                          color: isSelected ? accentColor : Colors.black12,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),

          // Signs list / grid
          Expanded(
            child: _isLoading
                ? const Center(
                    child: CircularProgressIndicator(color: accentColor),
                  )
                : _errorMessage != null
                    ? Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.error_outline,
                                  size: 40, color: Colors.redAccent),
                              const SizedBox(height: 12),
                              Text(_errorMessage!,
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(fontSize: 13)),
                              const SizedBox(height: 16),
                              ElevatedButton(
                                onPressed: _loadData,
                                child: const Text('Retry'),
                              ),
                            ],
                          ),
                        ),
                      )
                    : signs.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.search_off_rounded,
                                    size: 48, color: Colors.black26),
                                const SizedBox(height: 12),
                                const Text('No matching signs found',
                                    style: TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.w600,
                                        color: primaryColor)),
                                const SizedBox(height: 4),
                                Text(
                                  _coreOnly
                                      ? 'Try turning off "Core 50 Only" to view all 263 signs.'
                                      : 'Try searching with another word.',
                                  style: const TextStyle(
                                      fontSize: 12, color: Colors.black45),
                                ),
                              ],
                            ),
                          )
                        : RefreshIndicator(
                            onRefresh: _loadData,
                            child: ListView.builder(
                              padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
                              itemCount: signs.length,
                              itemBuilder: (context, idx) {
                                final s = signs[idx];
                                final label = s['label']?.toString() ?? 'Sign';
                                final cat = s['category']?.toString() ?? 'General';
                                final isCore = s['is_core_50'] == true;

                                return Container(
                                  margin: const EdgeInsets.only(bottom: 10),
                                  decoration: BoxDecoration(
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(16),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withValues(alpha: 0.04),
                                        blurRadius: 8,
                                        offset: const Offset(0, 2),
                                      ),
                                    ],
                                  ),
                                  child: ListTile(
                                    contentPadding: const EdgeInsets.symmetric(
                                        horizontal: 16, vertical: 6),
                                    onTap: () => _showSignDetails(s),
                                    leading: CircleAvatar(
                                      radius: 20,
                                      backgroundColor: isCore
                                          ? const Color(0xFFE6FAF3)
                                          : const Color(0xFFEEF2FF),
                                      child: Text(
                                        label.isNotEmpty
                                            ? label[0].toUpperCase()
                                            : '?',
                                        style: TextStyle(
                                          fontWeight: FontWeight.bold,
                                          fontSize: 16,
                                          color: isCore
                                              ? const Color(0xFF0F6E56)
                                              : accentColor,
                                        ),
                                      ),
                                    ),
                                    title: Text(
                                      label,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w600,
                                        fontSize: 15,
                                        color: primaryColor,
                                      ),
                                    ),
                                    subtitle: Row(
                                      children: [
                                        Text(
                                          cat,
                                          style: const TextStyle(
                                              fontSize: 12, color: Colors.black45),
                                        ),
                                        if (isCore) ...[
                                          const SizedBox(width: 8),
                                          Container(
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 6, vertical: 1),
                                            decoration: BoxDecoration(
                                              color: const Color(0xFFE6FAF3),
                                              borderRadius:
                                                  BorderRadius.circular(4),
                                            ),
                                            child: const Text(
                                              'Core 50',
                                              style: TextStyle(
                                                fontSize: 10,
                                                fontWeight: FontWeight.bold,
                                                color: Color(0xFF0F6E56),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ],
                                    ),
                                    trailing: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        IconButton(
                                          icon: const Icon(
                                              Icons.videocam_outlined,
                                              color: accentColor,
                                              size: 22),
                                          tooltip: 'Practice in camera',
                                          onPressed: () {
                                            Navigator.of(context).push(
                                              MaterialPageRoute(
                                                builder: (_) =>
                                                    const TranslateScreen(),
                                              ),
                                            );
                                          },
                                        ),
                                        const Icon(
                                          Icons.chevron_right_rounded,
                                          color: Colors.black26,
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
          ),
        ],
      ),
    );
  }
}
