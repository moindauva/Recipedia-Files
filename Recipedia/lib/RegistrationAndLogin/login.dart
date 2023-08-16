// ignore_for_file: use_build_context_synchronously

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:recipedia/RegistrationAndLogin/signup.dart';
import '../Admin/admin_portal.dart';
import '../Tabs/home_screen.dart';
import '../WidgetsAndUtils/progress_bar.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import '../WidgetsAndUtils/toast_message.dart';
import '../WidgetsAndUtils/user_model.dart';
import 'forget_password.dart';

class Login extends StatefulWidget {
  const Login({Key? key}) : super(key: key);

  @override
  State<Login> createState() => _LoginState();
}

class _LoginState extends State<Login> {
  final TextEditingController emailTextController = TextEditingController();
  final TextEditingController passwordTextController = TextEditingController();
  final firebaseAuth = FirebaseAuth.instance;
  BuildContext? dialogContext;

  Widget buildTextField(String labelText, String hintText,
      TextEditingController textController, bool isPasswordTextField) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 25.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.black45,
            ),
          ),
          TextField(
            obscureText: isPasswordTextField ? true : false,
            enabled: true,
            decoration: InputDecoration(
              hintText: hintText,
              hintStyle: const TextStyle(
                fontWeight: FontWeight.normal,
                color: Colors.black87,
                fontSize: 18,
              ),
              filled: false,
              floatingLabelBehavior: FloatingLabelBehavior.always,
              fillColor: Colors.black45,
            ),
            style: const TextStyle(color: Colors.black, height: 0.75),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.emailAddress,
            textInputAction: isPasswordTextField
                ? TextInputAction.done
                : TextInputAction.next,
          ),
        ],
      ),
    );
  }

  void login() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) {
          dialogContext = context;
          return ProgressBar(
            message: "Loading..",
          );
        },
      );
    });

    final user = (await firebaseAuth
            .signInWithEmailAndPassword(
                email: emailTextController.text,
                password: passwordTextController.text)
            .catchError(
      (errMsg) {
        if (errMsg.code == "user-not-found") {
          displayToastMessage("Login details are incorrect", context);
          Navigator.pop(dialogContext!);
        } else if (errMsg.code == "wrong-password") {
          displayToastMessage("Password is wrong", context);
          Navigator.pop(dialogContext!);
        } else if (errMsg.code == "invalid-email") {
          displayToastMessage("Email format is not valid", context);
          Navigator.pop(dialogContext!);
        } else if (errMsg.code == "too-many-requests") {
          displayToastMessage(
              "Too many failed attempts. Try again later.", context);
          Navigator.pop(dialogContext!);
        } else {
          displayToastMessage("Error: $errMsg", context);
          Navigator.pop(dialogContext!);
        }
      },
    ))
        .user;

    if (user != null) {
      if ('fyp.recipedia@gmail.com' == emailTextController.text &&
          'recipedia@admin' == passwordTextController.text) {
        Navigator.pop(dialogContext!);
        //To admin portal
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
              return AdminPortal();
            },
          ),
          (route) => false,
        );
      } else {
        String? userID = await UserModel().getUserID(emailTextController.text);
        String? name = await UserModel().getUserData(userID!, 'name');
        String? profilePicture =
            await UserModel().getUserData(userID, 'profilePictureURL');
        SharedPreference().saveCred(
          userID: userID,
          email: emailTextController.text,
          password: passwordTextController.text,
          name: name!,
          profilePicture: profilePicture!,
        );
        Navigator.pop(dialogContext!);
        //To home-screen
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
              return const HomeScreen(

              );
            },
          ),
          (route) => false,
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Stack(
          children: [
            Positioned.fill(
              child: Image.asset(
                'assets/login.png',
                fit: BoxFit.cover,
              ),
            ),
            Column(
              children: [
                SizedBox(
                  height: size.height * 0.45,
                ),
                Column(
                  mainAxisAlignment: MainAxisAlignment.start,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(
                        left: 25.0,
                        right: 25.0,
                      ),
                      //Welcome Back Text
                      child: Text(
                        'LOGIN',
                        textAlign: TextAlign.left,
                        style: TextStyle(
                          fontSize: 30,
                          fontWeight: FontWeight.bold,
                          color: Color(0XFFFF4F5A),
                        ),
                      ),
                    ),
                    SizedBox(
                      height: size.height * 0.04,
                    ),
                    Padding(
                      padding: const EdgeInsets.only(left: 25.0, right: 25.0),
                      child: Column(
                        children: [
                          //emailField,
                          buildTextField('EMAIL ADDRESS', 'username@gmail.com',
                              emailTextController, false),
                          //const SizedBox(height: 20.0),
                          //passwordField,
                          buildTextField('PASSWORD', '******',
                              passwordTextController, true),
                          SizedBox(
                            height: size.height * 0.010,
                          ),
                          //Forget Password (ROW)
                          Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                GestureDetector(
                                  onTap: () {
                                    Navigator.push(
                                      context,
                                      PageRouteBuilder(
                                        transitionDuration:
                                            const Duration(seconds: 1),
                                        transitionsBuilder: (context, animation,
                                            animationTime, child) {
                                          animation = CurvedAnimation(
                                              parent: animation,
                                              curve: Curves
                                                  .fastLinearToSlowEaseIn);
                                          return ScaleTransition(
                                            scale: animation,
                                            alignment: Alignment.center,
                                            child: child,
                                          );
                                        },
                                        pageBuilder: (context, animation,
                                            animationTime) {
                                          return const ForgetPassword();
                                        },
                                      ),
                                    );
                                  },
                                  child: const Text(
                                    "Forget Password",
                                    style: TextStyle(
                                        color: Color(0XFFFF4F5A),
                                        fontWeight: FontWeight.w400,
                                        fontSize: 18),
                                  ),
                                ),
                              ]),
                          SizedBox(
                            height: size.height * 0.010,
                          ),
                          //Login Button
                          Container(
                            width: MediaQuery.of(context).size.width,
                            height: 40,
                            margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                            decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(90)),
                            child: ElevatedButton(
                              onPressed: () {
                                if (emailTextController.text.isEmpty) {
                                  displayToastMessage(
                                      "Please enter email", context);
                                } else if (!emailTextController.text
                                    .contains('@')) {
                                  displayToastMessage(
                                      "Email is not valid", context);
                                } else if (!emailTextController.text
                                    .contains('.')) {
                                  displayToastMessage(
                                      "Please enter a valid email", context);
                                } else if (passwordTextController
                                    .text.isEmpty) {
                                  displayToastMessage(
                                      "Please enter password", context);
                                } else if (passwordTextController.text.length <
                                    6) {
                                  displayToastMessage(
                                      "Password must be atleast 6 characters",
                                      context);
                                } else {
                                  login();
                                }
                              },
                              style: ButtonStyle(
                                  backgroundColor:
                                      MaterialStateProperty.resolveWith(
                                          (states) {
                                    if (states
                                        .contains(MaterialState.pressed)) {
                                      return Colors.blueGrey;
                                    }
                                    return const Color(0XFFFF4F5A);
                                  }),
                                  shape: MaterialStateProperty.all(
                                      RoundedRectangleBorder(
                                          borderRadius:
                                              BorderRadius.circular(30)))),
                              child: const Text(
                                'LOG IN',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      height: size.height * 0.015,
                    ),
                    //Don't have an account (ROW)
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text(
                          "Don't have account? ",
                          style: TextStyle(color: Colors.black54, fontSize: 16),
                        ),
                        GestureDetector(
                          onTap: () {
                            Navigator.pushReplacement(
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
                                pageBuilder:
                                    (context, animation, animationTime) {
                                  return const Signup();
                                },
                              ),
                            );
                          },
                          child: const Text(
                            "Create new account.",
                            style: TextStyle(
                              color: Color(0XFFFF4F5A),
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                        )
                      ],
                    ),
                  ],
                ),
                SizedBox(
                  height: size.height * 0.025,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
