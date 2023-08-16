import 'package:flutter/material.dart';
import '../WidgetsAndUtils/snack_bar.dart';
import '../WidgetsAndUtils/toast_message.dart';
import '../WidgetsAndUtils/user_model.dart';
import 'login.dart';
import 'otp_screen.dart';

class Signup extends StatefulWidget {
  const Signup({Key? key}) : super(key: key);

  @override
  State<Signup> createState() => _SignupState();
}

class _SignupState extends State<Signup> {
  final TextEditingController nameTextController = TextEditingController();
  final TextEditingController emailTextController = TextEditingController();
  final TextEditingController passwordTextController = TextEditingController();

  //String email = '';
  late bool emailExists;
  //String password = '';
  //String name = '';

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
          TextFormField(
            decoration: InputDecoration(
              hintText: hintText,
              hintStyle: const TextStyle(color: Colors.black87, fontSize: 16),
            ),
            style: const TextStyle(color: Colors.black87),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.emailAddress,
            obscureText: isPasswordTextField ? true : false,
            textInputAction: isPasswordTextField
                ? TextInputAction.done
                : TextInputAction.next,
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      body: SingleChildScrollView(
          child: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/signup.png',
              fit: BoxFit.cover,
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              SizedBox(
                height: size.height * 0.35,
              ),
              //Create Account Heading
              const Padding(
                padding: EdgeInsets.only(
                  right: 25,
                  left: 25,
                ),
                child: Text(
                  'SIGN UP',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontSize: 35,
                      fontWeight: FontWeight.bold,
                      color: Color(0XFFFF4F5A)),
                ),
              ),
              SizedBox(
                height: size.height * 0.04,
              ),
              //Content
              Padding(
                  padding: const EdgeInsets.only(
                    left: 25.0,
                    right: 25.0,
                  ),
                  child: Column(
                    children: [
                      buildTextField(
                          'NAME', "User's Name", nameTextController, false),
                      buildTextField('EMAIL ADDRESS', "username@gmail.com",
                          emailTextController, false),
                      buildTextField(
                          'PASSWORD', "******", passwordTextController, true),
                      //Signup Button
                      Container(
                        width: MediaQuery.of(context).size.width,
                        height: 40,
                        margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(90)),
                        child: ElevatedButton(
                          onPressed: () async {
                            if (nameTextController.text.isEmpty) {
                              displayToastMessage("Please enter name", context);
                            } else if (nameTextController.text.length < 3) {
                              displayToastMessage(
                                  "Name must be atleast 3 characters", context);
                            } else if (nameTextController.text
                                .contains(RegExp(r'[0-9]'))) {
                              displayToastMessage(
                                  "Numbers and special characters cannot be included",
                                  context);
                            } else if (emailTextController.text.isEmpty) {
                              displayToastMessage(
                                  "Please enter email", context);
                            } else if (!emailTextController.text
                                .contains('@')) {
                              displayToastMessage(
                                  "Please enter a valid email", context);
                            } else if (!emailTextController.text
                                .contains('.')) {
                              displayToastMessage(
                                  "Please enter a valid email", context);
                            } else if (passwordTextController.text.isEmpty) {
                              displayToastMessage(
                                  "Please enter password", context);
                            } else if (passwordTextController.text.length < 6) {
                              displayToastMessage(
                                  "Password must be at-least 6 Characters",
                                  context);
                            } else {
                              emailExists = await UserModel()
                                  .checkIfEmailExists(emailTextController.text);
                              if (emailExists == true) {
                                // ignore: use_build_context_synchronously
                                snackBar(
                                    context, 'Email is already registered');
                              } else {
                                // ignore: use_build_context_synchronously
                                Navigator.push(
                                  context,
                                  PageRouteBuilder(
                                    transitionDuration:
                                        const Duration(seconds: 1),
                                    transitionsBuilder: (context, animation,
                                        animationTime, child) {
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
                                      return OTPScreen(
                                        email: emailTextController.text,
                                        password: passwordTextController.text,
                                        name: nameTextController.text,
                                      );
                                    },
                                  ),
                                );
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
                                      borderRadius:
                                          BorderRadius.circular(30)))),
                          child: const Text('SIGN UP',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16)),
                        ),
                      ),
                      SizedBox(
                        height: size.height * 0.015,
                      ),
                      //Already have an account? (ROW)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Text(
                            "Already have account? ",
                            style:
                                TextStyle(color: Colors.black54, fontSize: 16),
                          ),
                          GestureDetector(
                            onTap: () {
                              Navigator.pushReplacement(
                                context,
                                PageRouteBuilder(
                                  transitionDuration:
                                      const Duration(seconds: 1),
                                  transitionsBuilder: (context, animation,
                                      animationTime, child) {
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
                              );
                            },
                            child: const Text(
                              "Login.",
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
                  )),
              SizedBox(
                height: size.height * 0.025,
              ),
            ],
          ),
        ],
      )),
    );
  }
}
