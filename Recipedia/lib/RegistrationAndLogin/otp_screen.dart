// ignore_for_file: use_build_context_synchronously

import 'dart:async';
import 'package:email_otp/email_otp.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:recipedia/RegistrationAndLogin/signup.dart';
import 'package:recipedia/WidgetsAndUtils/notification.dart';
import '../Tabs/home_screen.dart';
import '../WidgetsAndUtils/progress_bar.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import '../WidgetsAndUtils/snack_bar.dart';
import '../WidgetsAndUtils/toast_message.dart';
import '../WidgetsAndUtils/user_model.dart';

class OTPScreen extends StatefulWidget {
  final String email;
  final String password;
  final String name;

  const OTPScreen({
    Key? key,
    required this.email,
    required this.password,
    required this.name,
  }) : super(key: key);

  @override
  State<OTPScreen> createState() => _OTPScreenState();
}

class _OTPScreenState extends State<OTPScreen> with TickerProviderStateMixin {
  final TextEditingController otpController = TextEditingController();
  EmailOTP emailAuth = EmailOTP();
  final firebaseAuth = FirebaseAuth.instance;
  late bool isButtonActive = true;
  int userID = 0;
  int userIDFromDatabase = 0;
  late Timer timer;
  int second = 60;

  void sendOTP(String userEmail) async {
    //EMAIL OTP
    emailAuth.setConfig(
        appEmail: "fyp.recipedia@gmail.com",
        appName: "Recipedia",
        userEmail: userEmail,
        otpLength: 6,
        otpType: OTPType.digitsOnly);
    if (await emailAuth.sendOTP() == true) {
      NotificationService().pushNotification(
          'We have sent you a OTP code. Please check your email.',
          'assets/OTP1.png');
      startTimer();
      snackBar(context, "OTP Sent");
    } else {
      snackBar(context, "Couldn't send OTP");
    }
  }

  Future<void> verifyOTP(String OTP) async {
    if (await emailAuth.verifyOTP(otp: OTP) == true) {
      snackBar(context, "OTP Verified");
      signup();
    } else {
      snackBar(context, "Invalid OTP Code");
    }
  }

  void signup() async {
    await firebaseAuth
        .createUserWithEmailAndPassword(
            email: widget.email, password: widget.password)
        .catchError(
      (errMsg) {
        if (errMsg.code == 'email-already-in-use') {
          displayToastMessage("Email already registered.", context);
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
                  return const Signup();
                },
              ),
              (route) => false);
        } else if (errMsg.code == 'invalid-email') {
          displayToastMessage("Email Format is Invalid", context);
        } else {
          displayToastMessage("Error: $errMsg", context);
        }
      },
    );
    ScaffoldMessenger.of(context).hideCurrentSnackBar();
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return ProgressBar(
          message: "Registering Please Wait..",
        );
      },
    );

    //Save user information into Database
    await UserModel().addUser(
        userID: userID,
        email: widget.email,
        name: widget.name,
        profilePictureURL:
            "https://static.vecteezy.com/system/resources/thumbnails/005/544/718/small/profile-icon-design-free-vector.jpg");
    SharedPreference().saveCred(
      userID: userID.toString(),
      email: widget.email,
      password: widget.password,
      name: widget.name,
      profilePicture:
          "https://static.vecteezy.com/system/resources/thumbnails/005/544/718/small/profile-icon-design-free-vector.jpg",
    );
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
        (route) => false);
  }

  void startTimer() {
    const oneSec = Duration(seconds: 1);
    timer = Timer.periodic(
      oneSec,
      (Timer timer) {
        if (second == 0) {
          setState(() {
            isButtonActive = true;
            timer.cancel();
          });
        } else {
          setState(() {
            second--;
          });
        }
      },
    );
  }

  @override
  void dispose() {
    timer.cancel();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    getUserData();
  }

  Future<void> getUserData() async {
    userIDFromDatabase = int.parse(await UserModel().getDocumentIds()) + 1;
    setState(() {
      userID = userIDFromDatabase;
    });
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
                'assets/OTP.png',
                fit: BoxFit.cover,
              ),
            ),
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  height: size.height * 0.475,
                ),
                Column(
                  children: [
                    //User Email Verification Text
                    const Text(
                      "User Email Verification",
                      style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 22.0,
                          color: Color(0XFFFF4F5A)),
                    ),
                    SizedBox(
                      height: size.height * 0.02,
                    ),
                    //Press Send OTP button to receive OTP code Text
                    const Text(
                      'Press Send OTP to receive OTP code',
                      style: TextStyle(
                        color: Colors.grey,
                        fontWeight: FontWeight.bold,
                        fontSize: 15.0,
                      ),
                    ),
                    SizedBox(
                      height: size.height * 0.01,
                    ),
                    //Email Text
                    Text(
                      widget.email,
                      style: const TextStyle(
                        color: Color(0XFFFF4F5A),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                SizedBox(
                  height: size.height * 0.02,
                ),
                Padding(
                  padding: const EdgeInsets.only(left: 20.0, right: 20.0),
                  child: Column(
                    children: [
                      //Code Input Field
                      Padding(
                        padding: const EdgeInsets.only(left: 20.0, right: 20),
                        child: Column(
                          children: [
                            TextField(
                              controller: otpController,
                              keyboardType: TextInputType.phone,
                              decoration: const InputDecoration(
                                border: InputBorder.none,
                                labelText: "Enter 6-digits code",
                                labelStyle: TextStyle(
                                  fontSize: 14.0,
                                  color: Colors.grey,
                                ),
                                focusedBorder: UnderlineInputBorder(
                                  borderSide: BorderSide(
                                    color: Color(0XFFFF6787),
                                    width: 2,
                                  ),
                                ),
                                enabledBorder: UnderlineInputBorder(
                                  borderSide: BorderSide(
                                    color: Colors.grey,
                                    width: 0.5,
                                  ),
                                ),
                                hintStyle: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 10.0,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(
                        height: size.height * 0.04,
                      ),
                      //Timer Text
                      RichText(
                        text: TextSpan(
                          children: [
                            const TextSpan(
                              text: "Click Send OTP again in ",
                              style: TextStyle(
                                  fontSize: 16.0,
                                  color: Colors.grey,
                                  fontWeight: FontWeight.bold),
                            ),
                            TextSpan(
                              text: "00:$second",
                              style: const TextStyle(
                                  fontSize: 16.0,
                                  color: Color(0XFFFF4F5A),
                                  fontWeight: FontWeight.bold),
                            ),
                            const TextSpan(
                              text: " Sec",
                              style: TextStyle(
                                  fontSize: 16.0,
                                  color: Colors.grey,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                      SizedBox(
                        height: size.height * 0.025,
                      ),
                      //Verify OTP Button
                      Container(
                        width: MediaQuery.of(context).size.width,
                        height: 40,
                        margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(90)),
                        child: ElevatedButton(
                          onPressed: () {
                            if (second == 0) {
                              setState(() {
                                timer.cancel();
                              });
                              displayToastMessage(
                                  "Your Code is Expire", context);
                            } else if (otpController.text.isEmpty) {
                              displayToastMessage("Enter OTP Code", context);
                            } else {
                              verifyOTP(otpController.text);
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
                          child: const Text('Verify OTP',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16)),
                        ),
                      ),
                      //Send OTP Button
                      Container(
                        width: MediaQuery.of(context).size.width,
                        height: 40,
                        margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(90),
                            border: Border.all(
                              color: const Color(0XFFFF4F5A),
                              style: BorderStyle.solid,
                              width: 2.5,
                            )),
                        child: ElevatedButton(
                          onPressed: () {
                            if (isButtonActive == true) {
                              if (second == 60) {
                                sendOTP(widget.email);

                                setState(() {
                                  isButtonActive = false;
                                });
                              } else {
                                snackBar(context,
                                    'Wait 60s before clicking the send OTP again.');
                              }
                            } else {}
                          },
                          style: ButtonStyle(
                              elevation:
                                  MaterialStateProperty.resolveWith<double>(
                                // As you said you don't need elevation. I'm returning 0 in both case
                                (Set<MaterialState> states) {
                                  if (states.contains(MaterialState.disabled)) {
                                    return 0;
                                  }
                                  return 0; // Defer to the widget's default.
                                },
                              ),
                              backgroundColor:
                                  MaterialStateProperty.resolveWith((states) {
                                if (states.contains(MaterialState.pressed)) {
                                  return Colors.blueGrey;
                                }
                                return Colors.transparent;
                              }),
                              shape: MaterialStateProperty.all(
                                  RoundedRectangleBorder(
                                      borderRadius:
                                          BorderRadius.circular(30)))),
                          child: const Text(
                            'Send OTP',
                            style: TextStyle(
                                color: Color(0XFFFF4F5A),
                                fontWeight: FontWeight.bold,
                                fontSize: 16),
                          ),
                        ),
                      )
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
