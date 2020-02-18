package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.ecore.EClass;

public class ExpressionImpl extends StatementImpl implements Expression {
	protected ExpressionImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.EXPRESSION;
	}

	public void evaluateStatement(State state) {
		((Expression) (this)).evaluateExpression((State) (state));
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = (Value) (null) ;
		return result;
	}
}
