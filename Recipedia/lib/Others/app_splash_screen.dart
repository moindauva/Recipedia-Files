import 'dart:async';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import '../RegistrationAndLogin/login_or_signup.dart';
import '../Tabs/home_screen.dart';
import '../WidgetsAndUtils/shared_preferences.dart';

class AppSplashScreen extends StatefulWidget {
  const AppSplashScreen({Key? key}) : super(key: key);

  @override
  State<AppSplashScreen> createState() => _AppSplashScreenState();
}

class _AppSplashScreenState extends State<AppSplashScreen> {
  FirebaseAuth firebaseAuth = FirebaseAuth.instance;
  String password = '';
  String email = '';

  void getDataAndCheck() async {
    bool emailPresent = await SharedPreference().checkValuePresent('email');
    bool passwordPresent =
        await SharedPreference().checkValuePresent('password');
    email = (await SharedPreference().getCred('email')) ?? '';
    password = (await SharedPreference().getCred('password')) ?? '';
    Timer(const Duration(seconds: 3), () {
      if (emailPresent == true && passwordPresent == true) {
        firebaseAuth
            .signInWithEmailAndPassword(email: email, password: password)
            .then((user) async {
          Navigator.pushAndRemoveUntil(
            context,
            PageRouteBuilder(
              transitionDuration: const Duration(seconds: 1),
              transitionsBuilder: (context, animation, animationTime, child) {
                animation = CurvedAnimation(
                    parent: animation, curve: Curves.fastLinearToSlowEaseIn);
                return ScaleTransition(
                  scale: animation,
                  alignment: Alignment.center,
                  child: child,
                );
              },
              pageBuilder: (context, animation, animationTime) {
                return const HomeScreen();
              },
            ),
            (route) => false,
          );
        }).catchError((e) {
          SharedPreference().reset();
          Navigator.pushAndRemoveUntil(
            context,
            PageRouteBuilder(
              transitionDuration: const Duration(seconds: 1),
              transitionsBuilder: (context, animation, animationTime, child) {
                animation = CurvedAnimation(
                    parent: animation, curve: Curves.fastLinearToSlowEaseIn);
                return ScaleTransition(
                  scale: animation,
                  alignment: Alignment.center,
                  child: child,
                );
              },
              pageBuilder: (context, animation, animationTime) {
                return const LoginOrSignUp();
              },
            ),
            (route) => false,
          );
        });
      } else {
        SharedPreference().reset();
        Navigator.pushAndRemoveUntil(
          context,
          PageRouteBuilder(
            transitionDuration: const Duration(seconds: 1),
            transitionsBuilder: (context, animation, animationTime, child) {
              animation = CurvedAnimation(
                  parent: animation, curve: Curves.fastLinearToSlowEaseIn);
              return ScaleTransition(
                scale: animation,
                alignment: Alignment.center,
                child: child,
              );
            },
            pageBuilder: (context, animation, animationTime) {
              return const LoginOrSignUp();
            },
          ),
          (route) => false,
        );
      }
    });
  }

  @override
  void initState() {
    super.initState();
    getDataAndCheck();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(color: Color(0XFFFF4F5A)
          /*gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0XFFFF1C2A),
              Color(0XFFFF4F5A),
            ]),*/
          ),
      child: Center(
          child: Image.asset(
        'assets/logo-white.png',
        height: 250.0,
        width: 250.0,
      )),
    );
  }
}
