package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class ExpressionImpl extends StatementImpl implements Expression {
	protected ExpressionImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.EXPRESSION;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Expression(this);
	}
}
