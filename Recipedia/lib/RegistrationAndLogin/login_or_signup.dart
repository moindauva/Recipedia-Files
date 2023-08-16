import 'package:flutter/material.dart';
import 'signup.dart';
import 'login.dart';

class LoginOrSignUp extends StatelessWidget {
  const LoginOrSignUp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/login-signup.gif',
              fit: BoxFit.cover,
            ),
          ),
          /*Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Colors.white.withOpacity(0.0), Colors.white],
              ),
            ),
          ),*/
          Column(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              SizedBox(
                height: size.height * 0.05,
              ),
              /*const Image(
              image: AssetImage("assets/login-signup.png"),
              width: 300.0,
              height: 300.0,
              alignment: Alignment.center,
            ),*/
              /*SizedBox(
              height: size.height * 0.025,
            ),*/
              //Recipedia Text with Slogan
              Container(
                alignment: Alignment.centerLeft,
                padding: const EdgeInsets.only(
                  left: 30,
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text(
                      'Cooking a\nDelicious Food\nEasily',
                      style: TextStyle(
                          color: Color(0XFFFF4F5A),
                          //color: Colors.white,
                          fontSize: 31.0,
                          fontWeight: FontWeight.w800,),
                    ),
                    /*SizedBox(
                      height: size.height * 0.015,
                    ),
                    const Text(
                      'Discover more than 2000 recipes\nin your hands and cooking it easily ',
                      style: TextStyle(
                          //color: Colors.black26,
                          color: Colors.white,
                          fontSize: 16.0,
                          fontWeight: FontWeight.bold,),
                    ),*/
                  ],
                ),
              ),
              SizedBox(
                height: size.height * 0.45,
              ),
              Container(
                alignment: Alignment.bottomCenter,
                padding: const EdgeInsets.only(left: 30.0, right: 30.0),
                child: Column(
                  children: [
                    //Login Button
                    Container(
                      width: MediaQuery.of(context).size.width,
                      height: 40,
                      margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                      decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(90)),
                      child: ElevatedButton(
                        onPressed: () {
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
                                return const Login();
                              },
                            ),
                          );
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
                        child: const Text(
                          'LOG IN',
                          style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 16),
                        ),
                      ),
                    ),
                    //Row (OR)
                    /*Row(
                      children: const [
                        Expanded(
                          child: Divider(
                            //color: Colors.black,
                            color: Colors.white,
                            thickness: 1,
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.all(8.0),
                          child: Text(
                            'OR',
                            style: TextStyle(
                              color: Colors.white,
                              //color: Colors.black87,
                            ),
                          ),
                        ),
                        Expanded(
                          child: Divider(
                            //color: Colors.black,
                            color: Colors.white,
                            thickness: 1,
                          ),
                        ),
                      ],
                    ),*/
                    SizedBox(
                      height: size.height * 0.02,
                    ),
                    //Signup Button
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
                                return const Signup();
                              },
                            ),
                          );
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
                                    borderRadius: BorderRadius.circular(30)))),
                        child: const Text(
                          'SIGN UP',
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
              SizedBox(
                height: size.height * 0.01,
              ),
            ],
          ),
        ],
      ),
    );
  }
}
