import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import '../WidgetsAndUtils/snack_bar.dart';
import '../WidgetsAndUtils/toast_message.dart';
import '../WidgetsAndUtils/user_model.dart';
import 'login.dart';

class ForgetPassword extends StatefulWidget {
  const ForgetPassword({super.key});

  @override
  State<ForgetPassword> createState() => _ForgetPasswordState();
}

class _ForgetPasswordState extends State<ForgetPassword> {
  TextEditingController emailTextController = TextEditingController();
  late bool emailExists;
  late String email;
  final firebaseAuth = FirebaseAuth.instance;

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
            textInputAction: TextInputAction.done,
          ),
        ],
      ),
    );
  }

  void resetPassword(BuildContext context) async {
    try {
      await firebaseAuth
          .sendPasswordResetEmail(email: emailTextController.text)
          .then(
        (value) {
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
                  return const Login();
                },
              ),
              (route) => false);
          snackBar(context, "Reset password email has been sent!");
        },
      );
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        snackBar(context, "User not exist");
      } else if (e.code == 'invalid-email') {
        snackBar(context, "Invalid email");
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
                'assets/forget-password.png',
                fit: BoxFit.cover,
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(
                left: 25.0,
                right: 25.0,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    height: size.height * 0.55,
                  ),
                  const Text(
                    'Forget\nPassword',
                    textAlign: TextAlign.left,
                    style: TextStyle(
                        fontSize: 35,
                        fontWeight: FontWeight.bold,
                        color: Color(0XFFFF4F5A)),
                  ),
                  SizedBox(
                    height: size.height * 0.04,
                  ),
                  buildTextField('EMAIL ADDRESS', 'username@gmail.com',
                      emailTextController, false),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    height: 40,
                    margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                    decoration:
                        BoxDecoration(borderRadius: BorderRadius.circular(90)),
                    child: ElevatedButton(
                      onPressed: () async {
                        if (emailTextController.text.isEmpty) {
                          displayToastMessage("Please enter email", context);
                        } else if (!emailTextController.text.contains('@')) {
                          displayToastMessage("Email is not valid", context);
                        } else if (!emailTextController.text.contains('.')) {
                          displayToastMessage(
                              "Please enter a valid email", context);
                        } else {
                          emailExists =
                              await UserModel().checkIfEmailExists(email);
                          if (emailExists == false) {
                            displayToastMessage(
                                "Email does not exist.", context);
                          } else {
                            // ignore: use_build_context_synchronously
                            resetPassword(context);
                          }
                        }
                      },
                      style: ButtonStyle(
                          backgroundColor:
                              MaterialStateProperty.resolveWith((states) {
                            if (states.contains(MaterialState.pressed)) {
                              return Colors.blueGrey;
                            }
                            return const Color(0XFFFF4F5A);
                          }),
                          shape: MaterialStateProperty.all(
                              RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30)))),
                      child: const Text('Forget Password',
                          style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 16)),
                    ),
                  ),
                  SizedBox(
                    height: size.height * 0.010,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      GestureDetector(
                        onTap: () {
                          Navigator.pushAndRemoveUntil(
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
                                  return const Login();
                                },
                              ),
                              (route) => false);
                        },
                        child: const Text(
                          "Login",
                          style: TextStyle(
                              color: Color(0XFFFF4F5A),
                              fontWeight: FontWeight.bold,
                              fontSize: 16),
                        ),
                      )
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
