package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Symbol;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class SymbolImpl extends TypedDeclarationImpl implements Symbol {
	protected SymbolImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SYMBOL;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Symbol(this);
	}
}
