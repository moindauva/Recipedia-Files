import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:recipedia/Admin/Feedback/feedback.dart';
import 'package:recipedia/Admin/Users/all_users.dart';
import 'package:recipedia/Admin/recipes/recipe_curd.dart';
import '../RegistrationAndLogin/login_or_signup.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import '../WidgetsAndUtils/toast_message.dart';

class AdminPortal extends StatelessWidget {
  AdminPortal({
    Key? key,
  }) : super(key: key);
  final firebaseAuth = FirebaseAuth.instance;

  Container button(BuildContext context, String title, Function onPressed) {
    return Container(
      width: MediaQuery.of(context).size.width * 0.85,
      height: 40,
      margin: const EdgeInsets.only(top: 10, bottom: 10),
      child: ElevatedButton(
        style: ButtonStyle(
          backgroundColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.pressed)) {
              return Colors.blueGrey;
            }
            return const Color(0XFFFF4F5A);
          }),
          shape: MaterialStateProperty.all(
            RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
          ),
        ),
        onPressed: () {
          onPressed();
        },
        child: Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ),
    );
  }

  Widget card(String text, String imagePath, Function onPressed) {
    return Padding(
        padding: const EdgeInsets.all(5.0),
        child: GestureDetector(
          onTap: () {
            onPressed();
          },
          child: Container(
            decoration: BoxDecoration(
                boxShadow: const [
                  BoxShadow(
                    //color: Colors.grey.withOpacity(0.5),
                    color: Colors.white,
                    //color: Color(0XFFD0D0D0),
                    spreadRadius: 0,
                    blurRadius: 0,
                  ),
                ],
                borderRadius: BorderRadius.circular(15),
                //color: const Color(0XFFFF6F6F6),
                color: Colors.white),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Image.asset(
                  imagePath,
                  height: 55,
                  width: 55,
                ),
                Text(
                  text,
                  style: const TextStyle(
                    color: Color(0XFFFF4F5A),
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                )
              ],
            ),
          ),
        ));
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            height: size.height * 0.05,
          ),
          const Image(
            image: AssetImage("assets/admin.png"),
            width: 300.0,
            height: 300.0,
            alignment: Alignment.center,
          ),
          SizedBox(
            height: size.height * 0.06,
          ),
          //Recipedia Text with Slogan
          Container(
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.only(
              left: 15,
              right: 15,
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.only(
                    left: 15,
                    right: 15,
                  ),
                  child: Column(
                    children: [
                      const Text(
                        'RECIPEDIA',
                        style: TextStyle(
                            color: Color(0XFFFF4F5A),
                            fontSize: 38.0,
                            fontWeight: FontWeight.w800),
                      ),
                      SizedBox(
                        height: size.height * 0.015,
                      ),
                      const Text(
                        'Aao Pakaaye milke!',
                        style: TextStyle(
                            color: Colors.black87,
                            fontSize: 22.0,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  height: size.height * 0.04,
                ),
                Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(15),
                    color: const Color(0XFFFF4F5A),
                  ),
                  child: GridView.count(
                    shrinkWrap: true,
                    crossAxisCount: 3,
                    mainAxisSpacing: 7.5,
                    crossAxisSpacing: 10,
                    padding: const EdgeInsets.all(10),
                    children: [
                      card('Recipes', 'assets/recipes1.png', () {
                        Navigator.push(
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
                              return RecipeCURD();
                            },
                          ),
                        );
                      }),
                      card('Users', 'assets/users.png', () {
                        Navigator.push(
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
                              return const AllUsers();
                            },
                          ),
                        );
                      }),
                      card('Feedback', 'assets/feedback.png', () {
                        Navigator.push(
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
                              return const RecipesFeedback();
                            },
                          ),
                        );
                      }),
                      card('Reports', 'assets/report.png', () {
                        displayToastMessage(
                            "Reports Container Pressed", context);
                      }),
                      card(
                        'Logout',
                        'assets/logout.png',
                        () {
                          SharedPreference().reset();
                          firebaseAuth.signOut().then(
                                (value) => Navigator.pushAndRemoveUntil(
                                    context,
                                    PageRouteBuilder(
                                      transitionDuration:
                                          const Duration(seconds: 1),
                                      transitionsBuilder: (context, animation,
                                          animationTime, child) {
                                        animation = CurvedAnimation(
                                            parent: animation,
                                            curve:
                                                Curves.fastLinearToSlowEaseIn);
                                        return ScaleTransition(
                                          scale: animation,
                                          alignment: Alignment.center,
                                          child: child,
                                        );
                                      },
                                      pageBuilder:
                                          (context, animation, animationTime) {
                                        return const LoginOrSignUp();
                                      },
                                    ),
                                    (route) => false),
                              );
                        },
                      )
                    ],
                  ),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
