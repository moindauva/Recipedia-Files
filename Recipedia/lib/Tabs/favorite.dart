import 'package:flutter/material.dart';
import 'package:recipedia/Admin/recipes/recipes.dart';
import '../Admin/recipes/recipe_card_slider.dart';
import '../Database/databaseHelper.dart';
import '../WidgetsAndUtils/shared_preferences.dart';
import 'profile.dart';
import 'setting.dart';
import 'home_screen.dart';

class Favorite extends StatefulWidget {
  const Favorite({
    Key? key,
  }) : super(key: key);

  @override
  State<Favorite> createState() => _FavoriteState();
}

class _FavoriteState extends State<Favorite> {
  List<Recipes> recipe = [];

  Future<List<Map<String, dynamic>>>
      getFavoriteRecipesDataFromLocalDatabase() async {
    String userID = (await SharedPreference().getCred('userID'))!;
    await DatabaseHelper().syncFavoriteRecipesDataFromFirestore(userID);

    List<Map<String, dynamic>> recipes =
        await DatabaseHelper().getAllRecipe('favoriteRecipeDataTable', 'All');
    //await Future.delayed(const Duration(seconds: 5));
    return recipes;
  }

  @override
  Widget build(BuildContext context) {
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
          'Favorite Recipes',
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
              children: [
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
                  onPressed: () {},
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
                              begin: const Offset(1.0, 0.0),
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
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      PageRouteBuilder(
                        transitionDuration: const Duration(milliseconds: 250),
                        transitionsBuilder:
                            (context, animation, secondaryAnimation, child) {
                          return SlideTransition(
                            position: Tween<Offset>(
                              begin: const Offset(1.0, 0.0),
                              end: Offset.zero,
                            ).animate(animation),
                            child: child,
                          );
                        },
                        pageBuilder: (context, animation, animationTime) {
                          return const Setting();
                        },
                      ),
                      (route) => false,
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
      body: Container(
        margin: const EdgeInsets.only(left: 15.0, right: 15.0),
        child: Center(
          child: FutureBuilder<List<Map<String, dynamic>>>(
            future: getFavoriteRecipesDataFromLocalDatabase(),
            builder: (BuildContext context,
                AsyncSnapshot<List<Map<String, dynamic>>> snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const CircularProgressIndicator();
              } else if (snapshot.hasError) {
                return Text('Error: ${snapshot.error}');
              } else {
                List<Map<String, dynamic>> recipes = snapshot.data!;
                return Column(
                  children: [
                    SizedBox(
                      height: size.height * 0.02,
                    ),
                    Expanded(
                      child: ListView.builder(
                        padding: const EdgeInsets.only(top: 0),
                        itemCount: recipes.length,
                        itemBuilder: (context, index) {
                          return Container(
                            margin: const EdgeInsets.symmetric(vertical: 5),
                            child: RecipeCardSlider(
                              isAdmin: false,
                              id: (recipes[index]['recipeId']).toString(),
                              title: recipes[index]['recipeName'],
                              description: recipes[index]['recipeDescription'],
                              cookTime: recipes[index]['recipeTime'],
                              course: recipes[index]['recipeCourse'],
                              diet: recipes[index]['recipeDiet'],
                              rating: recipes[index]['recipeRating'],
                              thumbnailUrl: recipes[index]['recipeURL'],
                              ingredients: recipes[index]['recipeIngredients'],
                              category: recipes[index]['recipeCategory'],
                            ),
                          );
                        },
                      ),
                    ),
                    /*Expanded(
                      child: ListView.builder(
                        padding: const EdgeInsets.only(top: 0),
                        itemCount: widget.favoriteRecipesToDisplay.length,
                        itemBuilder: (context, index) {
                          return RecipeCard(
                            favoriteRecipes: widget.favoriteRecipesToDisplay,
                            isAdmin: false,
                            ID: widget.favoriteRecipesToDisplay[index].recipeID,
                            title: widget.favoriteRecipesToDisplay[index].recipeName,
                            description: widget
                                .favoriteRecipesToDisplay[index].recipeDescription,
                            cookTime: widget.favoriteRecipesToDisplay[index].recipeTime,
                            rating: widget.favoriteRecipesToDisplay[index].recipeRating,
                            thumbnailUrl:
                            widget.favoriteRecipesToDisplay[index].recipeURL,
                            ingredients: widget
                                .favoriteRecipesToDisplay[index].recipeIngredients,
                          );
                        },
                      ),
                    )*/
                  ],
                );
              }
            },
          ),
        ),
      ),
    );
  }
}
