import 'package:flutter/material.dart';
import 'package:recipedia/WidgetsAndUtils/rating_model.dart';

class RecipesFeedback extends StatefulWidget {
  const RecipesFeedback({Key? key}) : super(key: key);

  @override
  State<RecipesFeedback> createState() => _RecipesFeedbackState();
}

class _RecipesFeedbackState extends State<RecipesFeedback> {
  Widget buildTextField(String comment) {
    final TextEditingController commentText = TextEditingController();
    commentText.text = comment;
    return Padding(
      padding: const EdgeInsets.only(
        bottom: 20.0,
      ),
      child: TextField(
        enabled: false,
        decoration: const InputDecoration(
          filled: false,
          floatingLabelBehavior: FloatingLabelBehavior.always,
          fillColor: Colors.black45,
        ),
        style: const TextStyle(
          fontSize: 15,
          color: Colors.black,
          height: 0.75,
        ),
        autofocus: false,
        controller: commentText,
        textInputAction: TextInputAction.done,
      ),
    );
  }

  Future<List<List<dynamic>>> getData() async {
    List<Map<String, Map<String, dynamic>>> feedbackList =
        await RatingModel().feedbackList();
    List<List<dynamic>> commentList = [];
    for (var item in feedbackList) {
      for (var key in item.keys) {
        var recipeId = key;
        var ratings = item[key]?['ratings'];
        var comments = item[key]?['comment'];
        var userIds = item[key]?['userID'];
        var usernames = item[key]?['username'];

        commentList.add([recipeId, ratings, comments, userIds, usernames]);
      }
    }
    return commentList;
  }

  @override
  Widget build(BuildContext context) {
    //Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Feedback'),
      ),
      body: FutureBuilder<List<List<dynamic>>>(
        future: getData(),
        builder: (BuildContext context,
            AsyncSnapshot<List<List<dynamic>>> snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const SizedBox(
              child: Center(
                child: CircularProgressIndicator(),
              ),
            );
          } else if (snapshot.hasError) {
            return Text('Error: ${snapshot.error}');
          } else {
            List<List<dynamic>> commentList = snapshot.data!;
            return ListView.builder(
              itemCount: commentList.length,
              itemBuilder: (BuildContext context, int index) {
                return ExpansionTile(
                  title: Text(
                    commentList[index][0],
                    style: const TextStyle(
                      fontSize: 16.0,
                      color: Colors.black87,
                      fontWeight: FontWeight.bold,
                      fontStyle: FontStyle.normal,
                    ),
                  ),
                  children: <Widget>[
                    Column(
                      children: <Widget>[
                        ListView.builder(
                          shrinkWrap: true,
                          itemCount: commentList[index][2].length,
                          itemBuilder: (BuildContext context, int innerIndex) {
                            return ListTile(
                              title: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    "${commentList[index][4][innerIndex]} (${commentList[index][3][innerIndex]})",
                                    style: const TextStyle(
                                      color: Color(0XFFFF4F5A),
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  Container(
                                    margin: const EdgeInsets.symmetric(
                                      vertical: 15,
                                    ),
                                    child: Text(
                                      "Stars Given: ${commentList[index][1][innerIndex]}",
                                      style: const TextStyle(
                                        fontSize: 16.0,
                                        color: Colors.black87,
                                        fontStyle: FontStyle.normal,
                                      ),
                                    ),
                                  ),
                                  buildTextField(
                                      commentList[index][2][innerIndex]),
                                ],
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ],
                );
              },
            );
          }
        },
      ),
    );
  }
}
