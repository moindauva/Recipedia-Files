import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'Database/databaseHelper.dart';
import 'Others/app_splash_screen.dart';
import 'WidgetsAndUtils/notification.dart';
import 'WidgetsAndUtils/shared_preferences.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();

  await SharedPreference().sharedPrefInit();

  NotificationService().initAwesomeNotification();
  NotificationService().requestPermission();

  DatabaseHelper dbHelper = DatabaseHelper();
  // initialize the local database
  await dbHelper.db;
  await dbHelper.syncDataFromFirestore();
  //await dbHelper.syncDataFromFirestore();
  //await dbHelper.syncFavoriteRecipesDataFromFirestore();

  MaterialColor myColor = const MaterialColor(0XFFFF4F5A, <int, Color>{
    50: Color(0XFFFF4F5A),
    100: Color(0XFFFF4F5A),
    200: Color(0XFFFF4F5A),
    300: Color(0XFFFF4F5A),
    400: Color(0XFFFF4F5A),
    500: Color(0XFFFF4F5A),
    600: Color(0XFFFF4F5A),
    700: Color(0XFFFF4F5A),
    800: Color(0XFFFF4F5A),
    900: Color(0XFFFF4F5A),
  });

  runApp(
    GetMaterialApp(
      theme: ThemeData(
        primarySwatch: myColor,
        fontFamily: 'Helvetica',
      ),
      debugShowCheckedModeBanner: false,
      initialRoute: 'AppSplashScreen',
      routes: {
        'AppSplashScreen': (context) => const AppSplashScreen(),
      },
    ),
  );
}
