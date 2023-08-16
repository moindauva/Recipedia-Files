// ignore_for_file: use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:recipedia/Admin/Recipes/review-rating.dart';
import 'package:share_plus/share_plus.dart';
import '../../Database/databaseHelper.dart';
import '../../WidgetsAndUtils/favorite_recipe_model.dart';
import '../../WidgetsAndUtils/shared_preferences.dart';
import '../../WidgetsAndUtils/toast_message.dart';
import 'add_recipe.dart';

class ViewRecipeDetails extends StatefulWidget {
  final String ID;
  final String title;
  final String description;
  final int rating;
  final String cookTime;
  final String thumbnailUrl;
  final String diet;
  final String course;
  String ingredients;

  ViewRecipeDetails({
    super.key,
    required this.ID,
    required this.title,
    required this.description,
    required this.rating,
    required this.cookTime,
    required this.thumbnailUrl,
    required this.diet,
    required this.course,
    required this.ingredients,
  });

  @override
  State<ViewRecipeDetails> createState() => _ViewRecipeDetailsState();
}

class _ViewRecipeDetailsState extends State<ViewRecipeDetails> {
  bool isExists = false;
  bool isFavoriteIcon = true;
  bool isFavorite = false;
  Icon favoriteIcon = const Icon(Icons.favorite_border_outlined);
  late String userID;
  late List recipeID = [widget.ID];
  late List<String> descriptionInBulletPoints = widget.description.split('. ');
  late List<String> ingredientsInList = widget.ingredients.split(', ');

  Future<void> loadData() async {
    userID = (await SharedPreference().getCred('userID'))!;
  }

  @override
  void initState() {
    loadData();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
        body: Stack(
      children: [
        SizedBox(
          width: size.width * 1,
          height: size.height * 0.4,
          child: Image.network(
            widget.thumbnailUrl,
            fit: BoxFit.cover,
          ),
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Column(children: [
              Container(
                margin: const EdgeInsets.only(left: 25, top: 25),
                height: 50,
                width: 50,
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: IconButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  alignment: Alignment.center,
                  icon: const Icon(
                    Icons.arrow_back_ios_new,
                    color: Color(0XFFFF4F5A),
                  ),
                ),
              ),
              Container(
                margin: const EdgeInsets.only(left: 25, top: 25),
                height: 50,
                width: 50,
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: IconButton(
                  onPressed: () async {
                    await Share.share(
                      '${widget.thumbnailUrl}\n\nCheck out this amazing recipe from Recipedia\n\n${widget.title}\n\nIngredients: ${widget.ingredients}\n\nSteps: ${widget.description}\n\nCategory: ${widget.course}\n\nDiet: ${widget.diet}\n\nTime: ${widget.cookTime}\n\nRating: ${widget.rating}' ,
                      subject: 'Check out this amazing recipe from Recipedia',
                    );
                  },
                  alignment: Alignment.center,
                  icon: const Icon(
                    Icons.share,
                    color: Color(0XFFFF4F5A),
                  ),
                ),
              ),
            ]),
            Column(
              children: [
                Container(
                  margin: const EdgeInsets.only(right: 25, top: 25),
                  height: 50,
                  width: 50,
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(30),
                  ),
                  child: IconButton(
                    onPressed: () async {
                      isExists = await DatabaseHelper().checkIfRecipeExists(
                        int.parse(userID),
                        int.parse(widget.ID),
                      );
                      if (isExists == true) {
                        displayToastMessage(
                            'Recipe is already in favorites', context);
                      } else {
                        DatabaseHelper().saveFavoriteRecipes(
                            int.parse(userID), int.parse(widget.ID));
                        FavoriteRecipeModel().addFavoriteRecipe(
                            userID: userID, recipeID: recipeID);
                        displayToastMessage(
                            'Recipe added to favorites', context);
                        setState(() {
                          if (isFavoriteIcon == true) {
                            isFavorite = true;
                            favoriteIcon = const Icon(Icons.favorite);
                            isFavoriteIcon = false;
                          } else {
                            favoriteIcon =
                                const Icon(Icons.favorite_border_outlined);
                            isFavoriteIcon = true;
                            isFavorite = false;
                          }
                        });
                      }
                    },
                    alignment: Alignment.center,
                    icon: favoriteIcon,
                    color: const Color(0XFFFF4F5A),
                  ),
                ),
                Container(
                  margin: const EdgeInsets.only(right: 25, top: 25),
                  height: 50,
                  width: 50,
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(30),
                  ),
                  child: IconButton(
                    onPressed: () async {
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
                            return ReviewAndRating(
                              id: widget.ID,
                              title: widget.title,
                              thumbnailUrl: widget.thumbnailUrl,
                            );
                          },
                        ),
                      );
                    },
                    alignment: Alignment.center,
                    icon: const Icon(Icons.comment),
                    color: const Color(0XFFFF4F5A),
                  ),
                ),
                Container(
                  margin: const EdgeInsets.only(right: 25, top: 25),
                  height: 50,
                  width: 50,
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(30),
                  ),
                  child: IconButton(
                    onPressed: () async {
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
                            return const AddRecipe();
                          },
                        ),
                      );
                    },
                    alignment: Alignment.center,
                    icon: const Icon(
                      Icons.add,
                    ),
                    color: const Color(0XFFFF4F5A),
                  ),
                ),
              ],
            )
          ],
        ),
        Container(
          margin: EdgeInsets.only(top: size.height * 0.35),
          width: size.width * 1,
          height: size.height * 0.65,
          decoration: const BoxDecoration(
            color: Color(0XFFF3F3F3),
            borderRadius: BorderRadius.only(
              topLeft: Radius.circular(45.0),
              topRight: Radius.circular(45.0),
            ),
          ),
          child: DefaultTabController(
            length: 2,
            child: Column(
              children: [
                SizedBox(
                  height: size.height * 0.03,
                ),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: size.width * 0.02),
                  child: Text(
                    widget.title,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ),
                SizedBox(
                  height: size.height * 0.03,
                ),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: size.width * 0.015),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Column(
                        children: [
                          Image.asset(
                            'assets/time.png',
                            width: 35,
                            height: 35,
                          ),
                          SizedBox(
                            height: size.height * 0.01,
                          ),
                          Text(
                            widget.cookTime,
                            style: const TextStyle(
                              color: Colors.black54,
                              fontSize: 16.0,
                            ),
                          ),
                          SizedBox(
                            height: size.height * 0.005,
                          ),
                          const Text(
                            'Cooking',
                            style: TextStyle(
                              color: Colors.black54,
                              fontSize: 14.0,
                            ),
                          ),
                        ],
                      ),
                      Column(
                        children: [
                          Image.asset(
                            'assets/rating.png',
                            width: 35,
                            height: 35,
                          ),
                          SizedBox(
                            height: size.height * 0.01,
                          ),
                          Text(
                            widget.rating.toString(),
                            style: const TextStyle(
                              color: Colors.black54,
                              fontSize: 16.0,
                            ),
                          ),
                          SizedBox(
                            height: size.height * 0.005,
                          ),
                          const Text(
                            'Ratting',
                            style: TextStyle(
                              color: Colors.black54,
                              fontSize: 14.0,
                            ),
                          ),
                        ],
                      ),
                      Column(
                        children: [
                          Image.asset(
                            'assets/diet.png',
                            width: 35,
                            height: 35,
                          ),
                          SizedBox(
                            height: size.height * 0.01,
                          ),
                          Text(
                            widget.diet,
                            style: const TextStyle(
                              color: Colors.black54,
                              fontSize: 16.0,
                            ),
                          ),
                          SizedBox(
                            height: size.height * 0.005,
                          ),
                          const Text(
                            'Diet',
                            style: TextStyle(
                              color: Colors.black54,
                              fontSize: 14.0,
                            ),
                          ),
                        ],
                      )
                    ],
                  ),
                ),
                SizedBox(
                  height: size.height * 0.03,
                ),
                //borderRadius: BorderRadius.circular(4),
                Container(
                  margin: EdgeInsets.symmetric(horizontal: size.width * 0.08),
                  decoration: BoxDecoration(
                    color: const Color(0XFFFF4F5A),
                    borderRadius:
                        BorderRadius.circular(40), // Add border radius
                  ),
                  child: ClipRRect(
                    // Add ClipRRect widget to prevent clipping of child widgets
                    borderRadius: BorderRadius.circular(40),
                    child: Container(
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            color: Color(0XFFFF4F5A),
                            width: 1.0,
                          ),
                        ),
                      ),
                      child: const TabBar(
                        labelColor: Colors.black,
                        unselectedLabelColor: Colors.white,
                        labelStyle: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                        indicatorSize: TabBarIndicatorSize.label,
                        indicatorColor: Colors.black,
                        indicatorPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                        ),
                        indicatorWeight: 2,
                        tabs: [
                          Tab(
                            child: Text(
                              'Instructions',
                              //style: TextStyle(color: Colors.white),
                            ),
                          ),
                          Tab(
                            child: Text(
                              'Ingredients',
                              //style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  height: size.height * 0.015,
                ),
                Expanded(
                  child: TabBarView(
                    children: [
                      Container(
                        margin: EdgeInsets.symmetric(
                          horizontal: size.width * 0.08,
                        ),
                        decoration: BoxDecoration(
                          border: Border.all(
                            width: 2,
                            color: const Color(0XFFFF4F5A),
                          ),
                          color: Colors.black87.withOpacity(0.75),
                          borderRadius: BorderRadius.circular(15),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withOpacity(0.5),
                              spreadRadius: 2,
                              blurRadius: 5,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.all(16.0),
                        child: ListView.builder(
                          shrinkWrap: true,
                          padding: EdgeInsets.zero,
                          itemCount: descriptionInBulletPoints.length,
                          itemBuilder: (context, index) {
                            return Container(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 10.0, horizontal: 16.0),
                              margin: const EdgeInsets.symmetric(
                                vertical: 3.5,
                              ),
                              child: Text(
                                "• ${descriptionInBulletPoints[index]}",
                                style: const TextStyle(
                                  color: Colors.white,
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                      Container(
                        margin: EdgeInsets.symmetric(
                          horizontal: size.width * 0.08,
                        ),
                        decoration: BoxDecoration(
                          border: Border.all(
                            width: 2,
                            color: const Color(0XFFFF4F5A),
                          ),
                          color: Colors.black87.withOpacity(0.75),
                          borderRadius: BorderRadius.circular(15),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withOpacity(0.5),
                              spreadRadius: 2,
                              blurRadius: 5,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.all(16.0),
                        //color: Colors.grey,
                        child: ListView.builder(
                          shrinkWrap: true,
                          padding: EdgeInsets.zero,
                          itemCount: ingredientsInList.length,
                          itemBuilder: (context, index) {
                            return Container(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 10.0, horizontal: 16.0),
                              margin: const EdgeInsets.symmetric(
                                vertical: 3.5,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0XFFFF4F5A),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.grey.withOpacity(0.3),
                                    spreadRadius: 1,
                                    blurRadius: 2,
                                    offset: const Offset(0, 1),
                                  ),
                                ],
                                borderRadius: BorderRadius.circular(8.0),
                              ),
                              child: Text(
                                "• ${ingredientsInList[index]}",
                                style: const TextStyle(
                                  color: Colors.white,
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                      /*Container(
                        margin: EdgeInsets.symmetric(
                          horizontal: size.width * 0.08,
                        ),
                        color: Colors.grey,
                        child: Center(
                          child: ListView.builder(
                              itemCount: widget.ingredients.length,
                              itemBuilder: (context, index) {
                                return Text(
                                  widget.ingredients[index],
                                  style: TextStyle(color: Colors.black),
                                );
                              }),
                        ),
                      ),*/
                    ],
                  ),
                ),
                SizedBox(
                  height: size.height * 0.03,
                ),
              ],
            ),
          ),
        ),
      ],
    ));
  }
}
/*return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Positioned(
            top: 0,
            right: 0,
            child: Column(
              children: [
                Container(
                  height: size.height * 0.35,
                  width: double.infinity, //size.width * 1,
                  color: Colors.white70,

                  child: Positioned.fill(
                    child: Image.network(
                      widget.thumbnailUrl,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),
                /*Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      margin: const EdgeInsets.only(left: 25, top: 25),
                      height: 50,
                      width: 50,
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.4),
                        borderRadius: BorderRadius.circular(30),
                      ),
                      child: IconButton(
                        onPressed: () {
                          Navigator.pop(context);
                        },
                        alignment: Alignment.center,
                        icon: const Icon(
                          Icons.arrow_back_ios_new,
                          color: Color(0XFFFF4F5A),
                        ),
                      ),
                    ),
                    Container(
                      margin: const EdgeInsets.only(right: 25, top: 25),
                      height: 50,
                      width: 50,
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.4),
                        borderRadius: BorderRadius.circular(30),
                      ),
                      child: IconButton(
                        onPressed: () async {
                          isExists = await DatabaseHelper()
                              .checkIfRecipeExists(
                                  int.parse(userID), int.parse(widget.ID));
                          if (isExists == true) {
                            displayToastMessage(
                                'Recipe is already in favorites', context);
                          } else {
                            DatabaseHelper().saveFavoriteRecipes(
                                int.parse(userID), int.parse(widget.ID));
                            FavoriteRecipeModel().addFavoriteRecipe(
                                userID: userID, recipeID: recipeID);
                            setState(() {
                              if (isFavoriteIcon == true) {
                                isFavorite = true;
                                favoriteIcon = const Icon(Icons.favorite);
                                isFavoriteIcon = false;
                              } else {
                                favoriteIcon = const Icon(
                                    Icons.favorite_border_outlined);
                                isFavoriteIcon = true;
                                isFavorite = false;
                              }
                            });
                          }
                        },
                        alignment: Alignment.center,
                        icon: favoriteIcon,
                        color: Color(0XFFFF4F5A),
                      ),
                    ),
                  ],
                ),*/
              ],
            ),
          ),
          Positioned(
            top: 100,
            right: 0,
            child: Container(
              alignment: Alignment.bottomCenter,
              height: size.height * 0.7,
              width: double.infinity, //size.width * 1,
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(40.0),
                  topRight: Radius.circular(40.0),
                ),
              ),
              child: Column(
                children: [
                  Text(
                    widget.title,
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.bold),
                  )
                ],
              ),
            ),
          ),
        ],
      ),
    );*/
