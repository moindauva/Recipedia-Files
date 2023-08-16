import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:recipedia/Others/FAQ.dart';
import 'package:recipedia/Others/about.dart';
import '../RegistrationAndLogin/login_or_signup.dart';
import '../WidgetsAndUtils/setting_tile.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import 'profile.dart';
import 'favorite.dart';
import 'home_screen.dart';

class Setting extends StatefulWidget {
  const Setting({
    Key? key,
  }) : super(key: key);

  @override
  State<Setting> createState() => _SettingState();
}

class _SettingState extends State<Setting> {
  final firebaseAuth = FirebaseAuth.instance;

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: Colors.white,
      body: Container(
        margin: const EdgeInsets.only(left: 20, right: 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: size.height * 0.05,
            ),
            const Text(
              "Recipedia",
              style: TextStyle(fontSize: 36, fontWeight: FontWeight.bold),
            ),
            SizedBox(
              height: size.height * 0.05,
            ),
            SettingsTile(
              color: const Color(0XFFFF4F5A),
              icon: Icons.person_2_outlined,
              title: "Account",
              onTap: () {
                Navigator.pushAndRemoveUntil(
                  context,
                  PageRouteBuilder(
                    transitionDuration: const Duration(milliseconds: 250),
                    transitionsBuilder:
                        (context, animation, secondaryAnimation, child) {
                      return SlideTransition(
                        position: Tween<Offset>(
                          begin: const Offset(-1.0, 0.0),
                          end: Offset.zero,
                        ).animate(animation),
                        child: child,
                      );
                    },
                    pageBuilder: (context, animation, animationTime) {
                      return const Profile();
                    },
                  ),
                  (route) => false,
                );
              },
            ),
            SizedBox(
              height: size.height * 0.025,
            ),
            SettingsTile(
              color: const Color(0XFFFF4F5A),
              icon: Icons.info_outline,
              title: "About Us",
              onTap: () {
                Navigator.push(
                  context,
                  PageRouteBuilder(
                    transitionDuration: const Duration(milliseconds: 250),
                    transitionsBuilder:
                        (context, animation, secondaryAnimation, child) {
                      return SlideTransition(
                        position: Tween<Offset>(
                          begin: const Offset(-1.0, 0.0),
                          end: Offset.zero,
                        ).animate(animation),
                        child: child,
                      );
                    },
                    pageBuilder: (context, animation, animationTime) {
                      return const About();
                    },
                  ),
                );
              },
            ),
            SizedBox(
              height: size.height * 0.025,
            ),
            SettingsTile(
              color: const Color(0XFFFF4F5A),
              icon: Icons.help_outline,
              title: "FAQs",
              onTap: () {
                Navigator.push(
                  context,
                  PageRouteBuilder(
                    transitionDuration: const Duration(milliseconds: 250),
                    transitionsBuilder:
                        (context, animation, secondaryAnimation, child) {
                      return SlideTransition(
                        position: Tween<Offset>(
                          begin: const Offset(-1.0, 0.0),
                          end: Offset.zero,
                        ).animate(animation),
                        child: child,
                      );
                    },
                    pageBuilder: (context, animation, animationTime) {
                      return FAQ();
                    },
                  ),
                );
              },
            ),
            SizedBox(
              height: size.height * 0.025,
            ),
            SettingsTile(
              color: const Color(0XFFFF4F5A),
              icon: Icons.logout,
              title: "Logout",
              onTap: () {
                SharedPreference().reset();
                firebaseAuth
                    .signOut()
                    .then((value) => Navigator.pushAndRemoveUntil(
                        context,
                        PageRouteBuilder(
                          transitionDuration: const Duration(seconds: 1),
                          transitionsBuilder:
                              (context, animation, animationTime, child) {
                            animation = CurvedAnimation(
                                parent: animation,
                                curve: Curves.fastLinearToSlowEaseIn);
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
                        (route) => false));
              },
            ),
          ],
        ),
      ),
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Settings',
          textAlign: TextAlign.center,
        ),
      ),
      resizeToAvoidBottomInset: false,
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.black87,
        onPressed: () {},
        child: const Icon(
          Icons.camera_alt,
          color: Colors.white,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: BottomAppBar(
        elevation: 0,
        color: Colors.transparent,
        shape: const CircularNotchedRectangle(),
        notchMargin: 5,
        clipBehavior: Clip.hardEdge,
        child: Container(
          height: kToolbarHeight,
          decoration: const BoxDecoration(
            color: Color(0XFFFF4F5A),
          ),
          child: Padding(
            padding: const EdgeInsets.only(left: 25, right: 25),
            child: Row(
              mainAxisSize: MainAxisSize.max,
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                IconButton(
                  icon: const Icon(
                    Icons.home,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(-1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const HomeScreen(
                          );
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
                IconButton(
                  icon: const Icon(
                    Icons.favorite_rounded,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(-1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const Favorite();
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
                const Padding(
                  padding: EdgeInsets.only(left: 25, right: 25),
                  child: SizedBox(),
                ),
                IconButton(
                  icon: const Icon(
                    Icons.person,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(-1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const Profile();
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
                IconButton(
                  icon: const Icon(
                    Icons.settings,
                    color: Colors.white,
                  ),
                  onPressed: () {},
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
