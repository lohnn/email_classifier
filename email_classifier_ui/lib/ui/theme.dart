import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static final darkTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: ColorScheme.fromSeed(
      seedColor: const Color(0xFF6C63FF), // Vibrant Purple
      brightness: Brightness.dark,
      surface: const Color(0xFF1E1E2E), // Dark Blue-Grey
      primary: const Color(0xFF6C63FF),
      secondary: const Color(0xFF00BFA6), // Teal accent
      tertiary: const Color(0xFFFF6584), // Pink accent
    ),
    scaffoldBackgroundColor: const Color(0xFF121218),
    cardTheme: CardThemeData(
      color: const Color(0xFF272736),
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    textTheme: GoogleFonts.outfitTextTheme(ThemeData.dark().textTheme),
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.transparent,
      elevation: 0,
      centerTitle: false,
    ),
  );
}
