// ignore_for_file: file_names, use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';
import 'package:recipedia/Tabs/home_screen.dart';
import 'package:recipedia/WidgetsAndUtils/shared_preferences.dart';
import 'package:recipedia/WidgetsAndUtils/toast_message.dart';
import '../../WidgetsAndUtils/rating_model.dart';
import '../../WidgetsAndUtils/user_model.dart';

class ReviewAndRating extends StatefulWidget {
  final String id;
  final String title;
  final String thumbnailUrl;

  const ReviewAndRating({
    Key? key,
    required this.id,
    required this.title,
    required this.thumbnailUrl,
  }) : super(key: key);

  @override
  State<ReviewAndRating> createState() => _ReviewAndRatingState();
}

class _ReviewAndRatingState extends State<ReviewAndRating> {
  TextEditingController recipeCommentTextController = TextEditingController();

  String recipeComment = '';
  String userName = '';
  String userID = '';
  int recipeRating = 0;

  void getData() async {
    String? email = await SharedPreference().getCred('email');
    String? id = await UserModel().getUserID(email!);
    String? name = await UserModel().getUserData(id!, 'name');

    setState(() {
      userName = name!;
      userID = id;
    });
  }

  bool containsString(String searchValue, List<String> list) {
    return list.contains(searchValue);
  }

  void recipeRatingFunction() async {
    bool ratingExist = await RatingModel().checkIfFeedbackExists(widget.id);
    List<String> id = await RatingModel().isUserIDExistsInFeedback(widget.id);
    bool isUserRatingExist = false;
    if (containsString(userID, id) == true) {
      isUserRatingExist = true;
    } else {
      isUserRatingExist = false;
    }
    if (ratingExist == true && isUserRatingExist == true) {
      displayToastMessage('You already gave the feedback.', context);
    } else if (ratingExist == true && isUserRatingExist == false) {
      RatingModel().updateRating(widget.id, recipeCommentTextController.text,
          userName, userID, recipeRating);
    } else if (ratingExist == false && isUserRatingExist == false) {
      RatingModel().addRating(widget.id, recipeCommentTextController.text,
          userName, userID, recipeRating);
    }

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

  @override
  void initState() {
    getData();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final recipeCommentField = TextField(
      maxLines: null,
      textAlign: TextAlign.start,
      decoration: InputDecoration(
        labelText: 'Enter your comment',
        labelStyle: const TextStyle(
          color: Colors.white,
          fontSize: 14.0,
        ),
        filled: true,
        floatingLabelBehavior: FloatingLabelBehavior.never,
        fillColor: Colors.black45,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15.0),
          borderSide: const BorderSide(width: 0, style: BorderStyle.none),
        ),
      ),
      style: const TextStyle(color: Colors.white, height: 1),
      autofocus: false,
      controller: recipeCommentTextController,
      keyboardType: TextInputType.multiline,
      onChanged: (text) {
        recipeComment = text;
      },
      textInputAction: TextInputAction.done,
    );

    Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Feedback',
          textAlign: TextAlign.center,
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.only(left: 15.0, right: 15.0, top: 30.0),
          child: Container(
            alignment: Alignment.center,
            child: Column(
              children: [
                Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    CircleAvatar(
                      backgroundImage: NetworkImage(widget.thumbnailUrl),
                      radius: 100.0,
                    ),
                    SizedBox(
                      height: size.height * 0.025,
                    ),
                    Text(
                      widget.title,
                      style: const TextStyle(
                        //color: Color(0XFFFF4F5A),
                        color: Colors.black,
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                SizedBox(
                  height: size.height * 0.05,
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'How would you like to rate the recipe?',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        //color: Color(0XFFFF4F5A),
                        color: Colors.black,
                        fontSize: 18,
                        fontWeight: FontWeight.normal,
                      ),
                    ),
                    SizedBox(
                      height: size.height * 0.025,
                    ),
                    RatingBar.builder(
                      initialRating: 0,
                      minRating: 1,
                      direction: Axis.horizontal,
                      allowHalfRating: true,
                      itemCount: 5,
                      itemPadding: const EdgeInsets.symmetric(horizontal: 4.0),
                      itemBuilder: (context, _) => const Icon(
                        Icons.star,
                        color: Colors.amber,
                      ),
                      onRatingUpdate: (rating) {
                        recipeRating = rating.toInt();
                      },
                    ),
                    SizedBox(
                      height: size.height * 0.05,
                    ),
                    const Text(
                      "We'd love to hear your thoughts on it. Did you enjoy the recipe?",
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        //color: Color(0XFFFF4F5A),
                        color: Colors.black,
                        fontSize: 18,
                        fontWeight: FontWeight.normal,
                      ),
                    ),
                    SizedBox(
                      height: size.height * 0.025,
                    ),
                    recipeCommentField,
                  ],
                ),
                SizedBox(
                  height: size.height * 0.01,
                ),
                Container(
                  width: MediaQuery.of(context).size.width,
                  height: 40,
                  margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
                  decoration:
                      BoxDecoration(borderRadius: BorderRadius.circular(90)),
                  child: ElevatedButton(
                    onPressed: () {
                      recipeRatingFunction();
                    },
                    style: ButtonStyle(
                        backgroundColor:
                            MaterialStateProperty.resolveWith((states) {
                          if (states.contains(MaterialState.pressed)) {
                            return Colors.blueGrey;
                          }
                          return const Color(0XFFFF4F5A);
                        }),
                        shape: MaterialStateProperty.all(RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30)))),
                    child: const Text(
                      'Send Feedback',
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
        ),
      ),
    );
  }
}
